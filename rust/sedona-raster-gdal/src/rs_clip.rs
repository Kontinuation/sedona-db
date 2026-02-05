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

use std::convert::TryFrom;
use std::sync::Arc;

use arrow_array::Array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use gdal::raster::{rasterize, Buffer, RasterizeOptions};
use gdal::vector::Geometry;
use gdal::DriverManager;

use arrow_schema::DataType;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::{nodata_bytes_to_f64, nodata_f64_to_bytes};
use crate::gdal_dataset_provider::configure_thread_local_cache_size;
use crate::raster_band_reader::RasterBandReader;

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
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        self.invoke_batch_from_args(arg_types, args, &SedonaType::Arrow(DataType::Null), 0, None)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        configure_thread_local_cache_size(config_options)?;
        let num_iterations = calc_num_iterations(args);

        // Parse arguments based on signature.
        // All supported variants put `raster` at index 0 and `geometry` at index 1 (no band)
        // or index 2 (with band).
        let geom_arg_idx = if self.with_band { 2 } else { 1 };

        // Expand band/nodata/all_touched to arrays so they can vary row-by-row.
        let band_array = if self.with_band {
            args[1]
                .clone()
                .cast_to(&arrow_schema::DataType::Int32, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Int32(Some(0)).to_array_of_size(num_iterations)?
        };
        let band_array = band_array
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| DataFusionError::Internal("Expected Int32Array for band".to_string()))?
            .clone();

        let nodata_array = if self.with_nodata {
            args[3]
                .clone()
                .cast_to(&arrow_schema::DataType::Float64, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Float64(None).to_array_of_size(num_iterations)?
        };
        let nodata_array = nodata_array
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .ok_or_else(|| {
                DataFusionError::Internal("Expected Float64Array for nodata".to_string())
            })?
            .clone();

        let all_touched_array = if self.with_all_touched {
            args[4]
                .clone()
                .cast_to(&arrow_schema::DataType::Boolean, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Boolean(Some(false)).to_array_of_size(num_iterations)?
        };
        let all_touched_array = all_touched_array
            .as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| {
                DataFusionError::Internal("Expected BooleanArray for allTouched".to_string())
            })?
            .clone();

        let mut band_iter = band_array.iter();
        let mut nodata_iter = nodata_array.iter();
        let mut all_touched_iter = all_touched_array.iter();

        // Build output rasters
        let mut builder = RasterBuilder::new(num_iterations);

        let exec_arg_types = vec![arg_types[0].clone(), arg_types[geom_arg_idx].clone()];
        let exec_args = vec![args[0].clone(), args[geom_arg_idx].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, geom_crs| {
            let band = band_iter.next().unwrap_or(Some(0)).unwrap_or(0);
            let nodata_value = nodata_iter.next().unwrap_or(None);
            let all_touched = all_touched_iter
                .next()
                .unwrap_or(Some(false))
                .unwrap_or(false);

            let (raster, geom_wkb) = match (raster_opt, wkb_opt) {
                (Some(r), Some(w)) => (r, w),
                _ => {
                    builder.append_null()?;
                    return Ok(());
                }
            };

            let raster_crs = raster.crs();
            let geom_wkb = if crate::crs_utils::crs_equivalent(raster_crs, geom_crs)? {
                geom_wkb.to_vec()
            } else {
                crate::crs_utils::transform_wkb_to_crs(geom_wkb, geom_crs, raster_crs)?
            };

            let band_index = usize::try_from(band.max(1)).unwrap_or(1);
            match clip_raster(raster, &geom_wkb, band_index, nodata_value, all_touched) {
                Ok(clipped_data) => build_clipped_raster(&mut builder, raster, &clipped_data)?,
                Err(e) => {
                    eprintln!("RS_Clip error: {}", e);
                    builder.append_null()?;
                }
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()?))
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
    let mut band_reader = RasterBandReader::new(raster);
    let width = metadata.width() as usize;
    let height = metadata.height() as usize;

    // Parse geometry from WKB
    let geometry = Geometry::from_wkb(geom_wkb).map_err(|e| {
        DataFusionError::Execution(format!("Failed to parse geometry from WKB: {}", e))
    })?;

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

    // Initialize mask to 0 (outside)
    let mut mask_band = mask_dataset
        .rasterband(1)
        .map_err(|e| DataFusionError::Execution(format!("Failed to get mask band: {}", e)))?;
    let zeros = vec![0u8; width * height];
    let mut buffer = Buffer::new((width, height), zeros);
    mask_band
        .write((0, 0), (width, height), &mut buffer)
        .map_err(|e| DataFusionError::Execution(format!("Failed to initialize mask: {}", e)))?;

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
        let original_data = band_reader.read_band_bytes(band_idx)?;

        // Determine nodata value
        let nodata = custom_nodata
            .or_else(|| nodata_bytes_to_f64(band_metadata.nodata_value(), &data_type))
            .unwrap_or(0.0);

        // Apply mask to band data
        let clipped_data =
            apply_mask_to_band(&original_data, mask, width, height, &data_type, nodata)?;

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

fn calc_num_iterations(args: &[ColumnarValue]) -> usize {
    for arg in args {
        if let ColumnarValue::Array(array) = arg {
            return array.len();
        }
    }
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::Edges;

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
        let mut reader = RasterBandReader::new(&raster);
        let original_len = reader.read_band_bytes(1).unwrap().len();
        assert_eq!(
            clipped.band_data[0].len(),
            original_len,
            "Clipped band should have same size as original"
        );
    }

    #[test]
    fn test_rs_clip_crs_mismatch() {
        use crate::rs_from_gdal_raster::RsFromGDALRaster;
        use sedona_expr::scalar_udf::SedonaScalarKernel;

        let probe = sedona_testing::create::make_wkb("POINT (0 0)");
        if let Err(err) =
            crate::crs_utils::transform_wkb_to_crs(&probe, Some("EPSG:4326"), Some("EPSG:3857"))
        {
            panic!("Unexpected CRS transform error: {}", err);
        }

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();

        let metadata = raster.metadata();
        let min_x = metadata.upper_left_x();
        let max_y = metadata.upper_left_y();
        let max_x = min_x + (metadata.width() as f64 * metadata.scale_x()) / 2.0;
        let min_y = max_y + (metadata.height() as f64 * metadata.scale_y()) / 2.0;

        let wkt = format!(
            "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
            min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
        );
        let geometry = Geometry::from_wkt(&wkt).unwrap();
        let geom_wkb = geometry.wkb().unwrap();

        // Generate the EPSG:3857 geometry using the same PROJ engine that the
        // UDF uses for CRS transforms. This makes the test robust to axis-order
        // and normalization differences between build configurations.
        let geom_wkb_merc =
            crate::crs_utils::transform_wkb_to_crs(&geom_wkb, Some("EPSG:4326"), Some("EPSG:3857"))
                .unwrap();

        let kernel = RsClip {
            with_band: false,
            with_nodata: false,
            with_all_touched: false,
        };

        let raster_scalar = ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_array)));
        let geom_type_4326 = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:4326").unwrap());
        let geom_type_3857 = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap());

        let result_4326 = kernel
            .invoke_batch(
                &[RASTER, geom_type_4326],
                &[
                    raster_scalar.clone(),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb))),
                ],
            )
            .unwrap();

        let result_3857 = kernel
            .invoke_batch(
                &[RASTER, geom_type_3857],
                &[
                    raster_scalar,
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb_merc))),
                ],
            )
            .unwrap();

        let band_data_4326 = match result_4326 {
            ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
                let array = RasterStructArray::new(struct_array.as_ref());
                let raster = array.get(0).unwrap();
                let data = raster.bands().band(1).unwrap().data().to_vec();
                data
            }
            _ => panic!("Expected raster scalar result"),
        };

        let band_data_3857 = match result_3857 {
            ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
                let array = RasterStructArray::new(struct_array.as_ref());
                let raster = array.get(0).unwrap();
                let data = raster.bands().band(1).unwrap().data().to_vec();
                data
            }
            _ => panic!("Expected raster scalar result"),
        };

        assert_eq!(band_data_4326, band_data_3857);
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
