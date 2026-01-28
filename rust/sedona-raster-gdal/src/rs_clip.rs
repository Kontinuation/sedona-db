// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! RS_Clip UDF - Clip a raster to a geometry boundary
//!
//! Similar to PostGIS ST_Clip, this function clips a raster to the extent of a geometry.
//! Pixels outside the geometry are set to nodata (or 0 if no nodata is defined).
//! The output raster has the same extent as the geometry's bounding box (within the
//! original raster bounds) with pixels outside the geometry masked.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, StructArray};
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use gdal::raster::{rasterize, Buffer, RasterizeOptions};
use gdal::vector::Geometry;
use gdal::DriverManager;

use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::{RasterRefImpl, RasterStructArray};
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::{nodata_bytes_to_f64, nodata_f64_to_bytes};

/// RS_Clip() scalar UDF implementation
///
/// Clips a raster to a geometry boundary
pub fn rs_clip_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_clip",
        vec![
            Arc::new(RsClip {
                with_band: false,
                with_nodata: false,
                with_all_touched: false,
            }),
            Arc::new(RsClip {
                with_band: true,
                with_nodata: false,
                with_all_touched: false,
            }),
            Arc::new(RsClip {
                with_band: true,
                with_nodata: true,
                with_all_touched: false,
            }),
            Arc::new(RsClip {
                with_band: true,
                with_nodata: true,
                with_all_touched: true,
            }),
        ],
        Volatility::Immutable,
        Some(rs_clip_doc()),
    )
}

fn rs_clip_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Clips a raster to the extent of a geometry. Pixels outside the geometry are set to nodata.".to_string(),
        "RS_Clip(raster: Raster, geometry: Geometry) or RS_Clip(raster: Raster, band: Integer, geometry: Geometry, nodata: Double, allTouched: Boolean)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster to clip")
    .with_argument("geometry", "Geometry: Clipping geometry (WKB format)")
    .with_argument("band", "Integer: Band number (1-based, optional - clips all bands if not specified)")
    .with_argument("nodata", "Double: Value to use for pixels outside the geometry (optional)")
    .with_argument("allTouched", "Boolean: If true, all pixels touched by the geometry are included (default: false)")
    .with_sql_example("SELECT RS_Clip(raster, ST_GeomFromText('POLYGON((...))')) FROM raster_table".to_string())
    .build()
}

/// Kernel implementation for RS_Clip
#[derive(Debug)]
struct RsClip {
    with_band: bool,
    with_nodata: bool,
    with_all_touched: bool,
}

impl SedonaScalarKernel for RsClip {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = if self.with_all_touched {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_boolean(),
            ]
        } else if self.with_nodata {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_numeric(),
            ]
        } else if self.with_band {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
            ]
        } else {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
            ]
        };

        let matcher = ArgMatcher::new(matchers, RASTER);
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let num_iterations = calc_num_iterations(args);

        // Parse arguments based on signature
        let (geom_arg_idx, band_num, nodata_value, all_touched) = if self.with_all_touched {
            let band = extract_i32_scalar(&args[1])?.unwrap_or(0);
            let nodata = extract_f64_scalar(&args[3])?;
            let all_touched = extract_bool_scalar(&args[4])?.unwrap_or(false);
            (2, band, nodata, all_touched)
        } else if self.with_nodata {
            let band = extract_i32_scalar(&args[1])?.unwrap_or(0);
            let nodata = extract_f64_scalar(&args[3])?;
            (2, band, nodata, false)
        } else if self.with_band {
            let band = extract_i32_scalar(&args[1])?.unwrap_or(0);
            (2, band, None, false)
        } else {
            (1, 0, None, false) // band=0 means all bands
        };

        // Get raster and geometry arrays
        let raster_array = get_raster_array(&args[0])?;
        let geom_array = get_binary_array(&args[geom_arg_idx])?;

        // Build output rasters
        let mut builder = RasterBuilder::new(num_iterations);

        for i in 0..num_iterations {
            let raster_idx = if raster_array.len() == 1 { 0 } else { i };
            let geom_idx = if geom_array.len() == 1 { 0 } else { i };

            if raster_array.is_null(raster_idx) || geom_array.is_null(geom_idx) {
                builder.append_null()?;
                continue;
            }

            let raster = raster_array.get(raster_idx)?;
            let geom_wkb = geom_array.value(geom_idx);

            match clip_raster(
                &raster,
                geom_wkb,
                band_num as usize,
                nodata_value,
                all_touched,
            ) {
                Ok(clipped_data) => {
                    build_clipped_raster(&mut builder, &raster, &clipped_data)?;
                }
                Err(e) => {
                    eprintln!("RS_Clip error: {}", e);
                    builder.append_null()?;
                }
            }
        }

        let result = Arc::new(builder.finish()?) as ArrayRef;
        finish_result(args, result)
    }
}

/// Data for a clipped raster
struct ClippedRasterData {
    /// Clipped band data (one Vec<u8> per band)
    band_data: Vec<Vec<u8>>,
    /// Band metadata (data types, nodata values)
    band_metadata: Vec<BandMetadata>,
}

/// Clip a raster to a geometry
fn clip_raster(
    raster: &RasterRefImpl<'_>,
    geom_wkb: &[u8],
    band_num: usize,
    custom_nodata: Option<f64>,
    all_touched: bool,
) -> Result<ClippedRasterData> {
    let metadata = raster.metadata();
    let bands = raster.bands();
    let width = metadata.width() as usize;
    let height = metadata.height() as usize;

    // Parse geometry from WKB
    let geometry = Geometry::from_wkb(geom_wkb).map_err(|e| {
        DataFusionError::Execution(format!("Failed to parse geometry from WKB: {}", e))
    })?;

    // Create GDAL dataset from raster to use for spatial reference (thread-local provider)
    let provider = crate::gdal_dataset_provider::thread_local_provider()
        .map_err(|e| DataFusionError::Execution(format!("Failed to init GDAL provider: {}", e)))?;
    let raster_ds = provider
        .raster_ref_to_gdal(raster)
        .map_err(|e| DataFusionError::Execution(format!("Failed to create GDAL dataset: {}", e)))?;
    let gdal_dataset = raster_ds.as_dataset();

    // Create a mask raster (same dimensions as input)
    let mem_driver = DriverManager::get_driver_by_name("MEM")
        .map_err(|e| DataFusionError::Execution(format!("Failed to get MEM driver: {}", e)))?;

    let mut mask_dataset = mem_driver
        .create_with_band_type::<u8, _>("", width, height, 1)
        .map_err(|e| DataFusionError::Execution(format!("Failed to create mask dataset: {}", e)))?;

    // Set the same geotransform as the input raster
    let geotransform = [
        metadata.upper_left_x(),
        metadata.scale_x(),
        metadata.skew_x(),
        metadata.upper_left_y(),
        metadata.skew_y(),
        metadata.scale_y(),
    ];
    mask_dataset
        .set_geo_transform(&geotransform)
        .map_err(|e| DataFusionError::Execution(format!("Failed to set geotransform: {}", e)))?;

    // Set spatial reference if available
    if let Ok(srs) = gdal_dataset.spatial_ref() {
        mask_dataset.set_spatial_ref(&srs).map_err(|e| {
            DataFusionError::Execution(format!("Failed to set spatial reference: {}", e))
        })?;
    }

    // Initialize mask to 0 (outside)
    {
        let mut mask_band = mask_dataset
            .rasterband(1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to get mask band: {}", e)))?;
        let zeros = vec![0u8; width * height];
        let mut buffer = Buffer::new((width, height), zeros);
        mask_band
            .write((0, 0), (width, height), &mut buffer)
            .map_err(|e| DataFusionError::Execution(format!("Failed to initialize mask: {}", e)))?;
    }

    // Rasterize geometry onto mask (set to 1 inside geometry)
    let rasterize_options = RasterizeOptions {
        all_touched,
        ..Default::default()
    };

    rasterize(
        &mut mask_dataset,
        &[1], // band 1
        &[geometry],
        &[1.0], // burn value = 1 (inside)
        Some(rasterize_options),
    )
    .map_err(|e| DataFusionError::Execution(format!("Failed to rasterize geometry: {}", e)))?;

    // Read the mask
    let mask_band = mask_dataset
        .rasterband(1)
        .map_err(|e| DataFusionError::Execution(format!("Failed to get mask band: {}", e)))?;
    let mask_buffer = mask_band
        .read_as::<u8>((0, 0), (width, height), (width, height), None)
        .map_err(|e| DataFusionError::Execution(format!("Failed to read mask: {}", e)))?;
    let mask = mask_buffer.data();

    // Determine which bands to process
    let band_indices: Vec<usize> = if band_num == 0 {
        (1..=bands.len()).collect()
    } else {
        if band_num > bands.len() {
            return Err(DataFusionError::Execution(format!(
                "Band {} is out of range (1-{})",
                band_num,
                bands.len()
            )));
        }
        vec![band_num]
    };

    // Process each band
    let mut clipped_band_data = Vec::new();
    let mut clipped_band_metadata = Vec::new();

    for &band_idx in &band_indices {
        let band = bands.band(band_idx).map_err(|e| {
            DataFusionError::Execution(format!("Failed to get band {}: {}", band_idx, e))
        })?;

        let band_metadata = band.metadata();
        let data_type = band_metadata.data_type();
        let original_data = band.data();

        // Determine nodata value
        let nodata = custom_nodata
            .or_else(|| nodata_bytes_to_f64(band_metadata.nodata_value(), &data_type))
            .unwrap_or(0.0);

        // Apply mask to band data
        let clipped_data =
            apply_mask_to_band(original_data, mask, width, height, &data_type, nodata)?;

        // Build band metadata
        let new_band_metadata = BandMetadata {
            nodata_value: Some(nodata_f64_to_bytes(nodata, &data_type)),
            storage_type: StorageType::InDb,
            datatype: data_type,
            outdb_url: None,
            outdb_band_id: None,
        };

        clipped_band_data.push(clipped_data);
        clipped_band_metadata.push(new_band_metadata);
    }

    Ok(ClippedRasterData {
        band_data: clipped_band_data,
        band_metadata: clipped_band_metadata,
    })
}

/// Apply mask to band data
fn apply_mask_to_band(
    original_data: &[u8],
    mask: &[u8],
    width: usize,
    height: usize,
    data_type: &BandDataType,
    nodata: f64,
) -> Result<Vec<u8>> {
    let byte_size = data_type_byte_size(data_type);
    let mut result = original_data.to_vec();

    for (pixel_idx, &mask_val) in mask.iter().enumerate().take(width * height) {
        if mask_val == 0 {
            // Pixel is outside geometry - set to nodata
            let byte_offset = pixel_idx * byte_size;
            write_nodata_value(&mut result, byte_offset, data_type, nodata)?;
        }
    }

    Ok(result)
}

/// Write nodata value to band data at specified offset
fn write_nodata_value(
    data: &mut [u8],
    offset: usize,
    data_type: &BandDataType,
    nodata: f64,
) -> Result<()> {
    match data_type {
        BandDataType::UInt8 => {
            data[offset] = nodata as u8;
        }
        BandDataType::UInt16 => {
            let bytes = (nodata as u16).to_le_bytes();
            data[offset..offset + 2].copy_from_slice(&bytes);
        }
        BandDataType::Int16 => {
            let bytes = (nodata as i16).to_le_bytes();
            data[offset..offset + 2].copy_from_slice(&bytes);
        }
        BandDataType::UInt32 => {
            let bytes = (nodata as u32).to_le_bytes();
            data[offset..offset + 4].copy_from_slice(&bytes);
        }
        BandDataType::Int32 => {
            let bytes = (nodata as i32).to_le_bytes();
            data[offset..offset + 4].copy_from_slice(&bytes);
        }
        BandDataType::Float32 => {
            let bytes = (nodata as f32).to_le_bytes();
            data[offset..offset + 4].copy_from_slice(&bytes);
        }
        BandDataType::Float64 => {
            let bytes = nodata.to_le_bytes();
            data[offset..offset + 8].copy_from_slice(&bytes);
        }
    }
    Ok(())
}

/// Build clipped raster using RasterBuilder
fn build_clipped_raster(
    builder: &mut RasterBuilder,
    original_raster: &RasterRefImpl<'_>,
    clipped_data: &ClippedRasterData,
) -> Result<()> {
    let original_metadata = original_raster.metadata();

    // Use original raster dimensions and geotransform
    let metadata = RasterMetadata {
        width: original_metadata.width(),
        height: original_metadata.height(),
        upperleft_x: original_metadata.upper_left_x(),
        upperleft_y: original_metadata.upper_left_y(),
        scale_x: original_metadata.scale_x(),
        scale_y: original_metadata.scale_y(),
        skew_x: original_metadata.skew_x(),
        skew_y: original_metadata.skew_y(),
    };

    builder
        .start_raster(&metadata, original_raster.crs())
        .map_err(|e| DataFusionError::Execution(format!("Failed to start raster: {}", e)))?;

    // Add clipped bands
    for (band_data, band_metadata) in clipped_data
        .band_data
        .iter()
        .zip(clipped_data.band_metadata.iter())
    {
        builder
            .start_band(band_metadata.clone())
            .map_err(|e| DataFusionError::Execution(format!("Failed to start band: {}", e)))?;
        builder.band_data_writer().append_value(band_data);
        builder
            .finish_band()
            .map_err(|e| DataFusionError::Execution(format!("Failed to finish band: {}", e)))?;
    }

    builder
        .finish_raster()
        .map_err(|e| DataFusionError::Execution(format!("Failed to finish raster: {}", e)))?;

    Ok(())
}

/// Get byte size of data type
fn data_type_byte_size(data_type: &BandDataType) -> usize {
    match data_type {
        BandDataType::UInt8 => 1,
        BandDataType::UInt16 | BandDataType::Int16 => 2,
        BandDataType::UInt32 | BandDataType::Int32 | BandDataType::Float32 => 4,
        BandDataType::Float64 => 8,
    }
}

/// Helper to get raster array from ColumnarValue
fn get_raster_array(arg: &ColumnarValue) -> Result<RasterStructArray<'_>> {
    match arg {
        ColumnarValue::Array(array) => {
            let struct_array = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal("Expected StructArray for raster".to_string())
                })?;
            Ok(RasterStructArray::new(struct_array))
        }
        ColumnarValue::Scalar(ScalarValue::Struct(arc_struct)) => {
            Ok(RasterStructArray::new(arc_struct.as_ref()))
        }
        _ => Err(DataFusionError::Internal(
            "Expected raster argument".to_string(),
        )),
    }
}

/// Helper to get binary array from ColumnarValue
fn get_binary_array(arg: &ColumnarValue) -> Result<Arc<BinaryArray>> {
    match arg {
        ColumnarValue::Array(array) => {
            let binary_array = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal("Expected BinaryArray for geometry".to_string())
                })?;
            Ok(Arc::new(binary_array.clone()))
        }
        ColumnarValue::Scalar(scalar) => {
            let array = scalar.to_array()?;
            let binary_array = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal("Expected BinaryArray for geometry".to_string())
                })?;
            Ok(Arc::new(binary_array.clone()))
        }
    }
}

/// Helper to extract i32 scalar value
fn extract_i32_scalar(arg: &ColumnarValue) -> Result<Option<i32>> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Int32(v)) => Ok(*v),
        ColumnarValue::Scalar(ScalarValue::Int64(v)) => Ok(v.map(|x| x as i32)),
        ColumnarValue::Scalar(ScalarValue::Int16(v)) => Ok(v.map(|x| x as i32)),
        ColumnarValue::Scalar(ScalarValue::Int8(v)) => Ok(v.map(|x| x as i32)),
        _ => Ok(None),
    }
}

/// Helper to extract f64 scalar value
fn extract_f64_scalar(arg: &ColumnarValue) -> Result<Option<f64>> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Float64(v)) => Ok(*v),
        ColumnarValue::Scalar(ScalarValue::Float32(v)) => Ok(v.map(|x| x as f64)),
        ColumnarValue::Scalar(ScalarValue::Int64(v)) => Ok(v.map(|x| x as f64)),
        ColumnarValue::Scalar(ScalarValue::Int32(v)) => Ok(v.map(|x| x as f64)),
        _ => Ok(None),
    }
}

/// Helper to extract bool scalar value
fn extract_bool_scalar(arg: &ColumnarValue) -> Result<Option<bool>> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Boolean(v)) => Ok(*v),
        _ => Ok(None),
    }
}

/// Calculate number of iterations
fn calc_num_iterations(args: &[ColumnarValue]) -> usize {
    for arg in args {
        if let ColumnarValue::Array(array) = arg {
            return array.len();
        }
    }
    1
}

/// Convert result to appropriate ColumnarValue
fn finish_result(args: &[ColumnarValue], out: ArrayRef) -> Result<ColumnarValue> {
    for arg in args {
        if let ColumnarValue::Array(_) = arg {
            return Ok(ColumnarValue::Array(out));
        }
    }
    Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
        out.as_ref(),
        0,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rs_clip_basic() {
        // Load test raster
        use crate::rs_from_gdal_raster::RsFromGDALRaster;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();

        // Create a simple polygon WKB that covers part of the raster
        // Get raster bounds first
        let metadata = raster.metadata();
        let min_x = metadata.upper_left_x();
        let max_y = metadata.upper_left_y();
        let max_x = min_x + (metadata.width() as f64 * metadata.scale_x()) / 2.0;
        let min_y = max_y + (metadata.height() as f64 * metadata.scale_y()) / 2.0;

        // Create a box polygon covering half the raster
        let wkt = format!(
            "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
            min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
        );

        let geometry = Geometry::from_wkt(&wkt).unwrap();
        let geom_wkb = geometry.wkb().unwrap();

        // Clip the raster
        let result = clip_raster(&raster, &geom_wkb, 0, None, false);
        assert!(result.is_ok(), "Clip should succeed: {:?}", result.err());

        let clipped = result.unwrap();
        assert!(
            !clipped.band_data.is_empty(),
            "Should have at least one band"
        );

        // Verify band data size matches original
        let original_band = raster.bands().band(1).unwrap();
        assert_eq!(
            clipped.band_data[0].len(),
            original_band.data().len(),
            "Clipped band should have same size as original"
        );
    }

    #[test]
    fn test_write_nodata_value() {
        let mut data = vec![0u8; 8];

        // Test UInt8
        write_nodata_value(&mut data, 0, &BandDataType::UInt8, 255.0).unwrap();
        assert_eq!(data[0], 255);

        // Test Float32
        write_nodata_value(&mut data, 0, &BandDataType::Float32, -9999.0).unwrap();
        let value = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert!((value - (-9999.0)).abs() < 0.001);
    }
}
