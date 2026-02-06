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

//! RS_AsRaster UDF - Rasterize a vector geometry onto a raster grid.
//!
//! RS_AsRaster converts a vector geometry into a raster dataset by assigning a
//! specified value to all pixels covered by the geometry.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, StructArray};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use gdal::raster::Buffer;
use gdal::vector::Geometry;
use gdal::DriverManager;

use arrow_schema::DataType;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::{RasterRefImpl, RasterStructArray};
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::nodata_f64_to_bytes;
use crate::gdal_dataset_provider::configure_thread_local_cache_size;
use crate::gdal_rasterize_affine::rasterize_affine;

/// RS_AsRaster() scalar UDF implementation
pub fn rs_as_raster_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_asraster",
        vec![
            Arc::new(RsAsRaster { arg_count: 3 }),
            Arc::new(RsAsRaster { arg_count: 4 }),
            Arc::new(RsAsRaster { arg_count: 5 }),
            Arc::new(RsAsRaster { arg_count: 6 }),
            Arc::new(RsAsRaster { arg_count: 7 }),
        ],
        Volatility::Immutable,
        Some(rs_as_raster_doc()),
    )
}

fn rs_as_raster_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Rasterizes a geometry onto the grid of a reference raster and returns a newly generated raster.".to_string(),
        "RS_AsRaster(geom: Geometry, raster: Raster, pixelType: String, allTouched: Boolean, value: Double, noDataValue: Double, useGeometryExtent: Boolean)".to_string(),
    )
    .with_argument("geom", "Geometry: Input geometry (WKB format)")
    .with_argument("raster", "Raster: Reference raster defining CRS and grid")
    .with_argument(
        "pixelType",
        "String: Output pixel type (D, F, I, S, US, B)",
    )
    .with_argument(
        "allTouched",
        "Boolean: If true, include all touched pixels (default: false)",
    )
    .with_argument(
        "value",
        "Double: Burn value to assign inside geometry (default: 1.0)",
    )
    .with_argument(
        "noDataValue",
        "Double: Output nodata value (default: null)",
    )
    .with_argument(
        "useGeometryExtent",
        "Boolean: If true, output extent is geometry envelope; if false, uses reference raster extent (default: true)",
    )
    .with_sql_example(
        "SELECT RS_AsRaster(ST_GeomFromWKT('POLYGON((15 15, 18 20, 15 24, 24 25, 15 15))'), RS_MakeEmptyRaster(2, 255, 255, 3, -215, 2, -2, 0, 0, 4326), 'D')"
            .to_string(),
    )
    .build()
}

#[derive(Debug)]
struct RsAsRaster {
    /// Number of arguments in the matched signature (3..=7)
    arg_count: usize,
}

impl SedonaScalarKernel for RsAsRaster {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![
            ArgMatcher::is_geometry_or_geography(),
            ArgMatcher::is_raster(),
            ArgMatcher::is_string(),
        ];

        if self.arg_count >= 4 {
            matchers.push(ArgMatcher::is_boolean());
        }
        if self.arg_count >= 5 {
            matchers.push(ArgMatcher::is_numeric());
        }
        if self.arg_count >= 6 {
            matchers.push(ArgMatcher::is_numeric());
        }
        if self.arg_count >= 7 {
            matchers.push(ArgMatcher::is_boolean());
        }

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
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        configure_thread_local_cache_size(config_options)?;
        let num_iterations = calc_num_iterations(args);

        // Required arguments
        let pixel_type_str = extract_string_scalar(&args[2])?.ok_or_else(|| {
            DataFusionError::Execution("pixelType is required and must be a string".to_string())
        })?;
        let band_type = parse_pixel_type(&pixel_type_str)?;

        // Optional args (defaults per spec)
        let all_touched = if self.arg_count >= 4 {
            extract_bool_scalar(&args[3])?.unwrap_or(false)
        } else {
            false
        };
        let burn_value = if self.arg_count >= 5 {
            extract_f64_scalar(&args[4])?.unwrap_or(1.0)
        } else {
            1.0
        };
        let nodata_value = if self.arg_count >= 6 {
            extract_f64_scalar(&args[5])?
        } else {
            None
        };
        let use_geometry_extent = if self.arg_count >= 7 {
            extract_bool_scalar(&args[6])?.unwrap_or(true)
        } else {
            true
        };

        let geom_array = get_binary_array(&args[0])?;
        let raster_array = get_raster_array(&args[1])?;

        let mut builder = RasterBuilder::new(num_iterations);

        for i in 0..num_iterations {
            let geom_idx = if geom_array.len() == 1 { 0 } else { i };
            let raster_idx = if raster_array.len() == 1 { 0 } else { i };

            if geom_array.is_null(geom_idx) || raster_array.is_null(raster_idx) {
                builder.append_null()?;
                continue;
            }

            let geom_wkb = geom_array.value(geom_idx);
            let raster = raster_array.get(raster_idx)?;

            match as_raster(
                geom_wkb,
                &raster,
                band_type,
                all_touched,
                burn_value,
                nodata_value,
                use_geometry_extent,
            ) {
                Ok((out_metadata, out_band_metadata, out_band_bytes)) => {
                    builder
                        .start_raster(&out_metadata, raster.crs())
                        .map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Failed to start output raster: {}",
                                e
                            ))
                        })?;

                    builder.start_band(out_band_metadata).map_err(|e| {
                        DataFusionError::Execution(format!(
                            "Failed to start output raster band: {}",
                            e
                        ))
                    })?;

                    builder.band_data_writer().append_value(&out_band_bytes);
                    builder.finish_band().map_err(|e| {
                        DataFusionError::Execution(format!(
                            "Failed to finish output raster band: {}",
                            e
                        ))
                    })?;

                    builder.finish_raster().map_err(|e| {
                        DataFusionError::Execution(format!("Failed to finish output raster: {}", e))
                    })?;
                }
                Err(e) => {
                    eprintln!("RS_AsRaster error: {}", e);
                    builder.append_null()?;
                }
            }
        }

        let result = Arc::new(builder.finish()?) as ArrayRef;
        finish_result(args, result)
    }
}

fn parse_pixel_type(s: &str) -> Result<BandDataType> {
    match s.trim().to_ascii_uppercase().as_str() {
        "D" => Ok(BandDataType::Float64),
        "F" => Ok(BandDataType::Float32),
        "I" => Ok(BandDataType::Int32),
        "S" => Ok(BandDataType::Int16),
        "US" => Ok(BandDataType::UInt16),
        "B" => Ok(BandDataType::UInt8),
        "I8" | "INT8" => Ok(BandDataType::Int8),
        "U64" | "UINT64" => Ok(BandDataType::UInt64),
        "I64" | "INT64" => Ok(BandDataType::Int64),
        other => Err(DataFusionError::Execution(format!(
            "Unsupported pixelType: {} (expected one of D, F, I, S, US, B, I8, U64, I64)",
            other
        ))),
    }
}

fn as_raster(
    geom_wkb: &[u8],
    reference_raster: &RasterRefImpl<'_>,
    band_type: BandDataType,
    all_touched: bool,
    burn_value: f64,
    nodata_value: Option<f64>,
    use_geometry_extent: bool,
) -> Result<(RasterMetadata, BandMetadata, Vec<u8>)> {
    let ref_md = reference_raster.metadata();

    if ref_md.skew_x() != 0.0 || ref_md.skew_y() != 0.0 {
        return Err(DataFusionError::Execution(
            "RS_AsRaster currently requires skew_x=0 and skew_y=0 in the reference raster"
                .to_string(),
        ));
    }

    // Parse geometry
    let geometry = Geometry::from_wkb(geom_wkb).map_err(|e| {
        DataFusionError::Execution(format!("Failed to parse geometry from WKB: {}", e))
    })?;

    // Compute output grid
    let (out_width, out_height, out_ulx, out_uly) = if use_geometry_extent {
        let env = geometry.envelope();
        let ulx = ref_md.upper_left_x();
        let uly = ref_md.upper_left_y();
        let scale_x = ref_md.scale_x();
        let scale_y = ref_md.scale_y();

        if scale_x == 0.0 || scale_y == 0.0 {
            return Err(DataFusionError::Execution(
                "Reference raster has zero scale".to_string(),
            ));
        }

        let start_col = ((env.MinX - ulx) / scale_x).floor() as isize;
        let end_col_excl = ((env.MaxX - ulx) / scale_x).ceil() as isize;

        // Note: scale_y is typically negative.
        let start_row = ((env.MaxY - uly) / scale_y).floor() as isize;
        let end_row_excl = ((env.MinY - uly) / scale_y).ceil() as isize;

        let width = (end_col_excl - start_col).max(0) as usize;
        let height = (end_row_excl - start_row).max(0) as usize;

        if width == 0 || height == 0 {
            return Err(DataFusionError::Execution(
                "Geometry extent produced an empty raster".to_string(),
            ));
        }

        let out_ulx = ulx + (start_col as f64) * scale_x;
        let out_uly = uly + (start_row as f64) * scale_y;
        (width, height, out_ulx, out_uly)
    } else {
        (
            ref_md.width() as usize,
            ref_md.height() as usize,
            ref_md.upper_left_x(),
            ref_md.upper_left_y(),
        )
    };

    // Create output GDAL dataset
    let mem_driver = DriverManager::get_driver_by_name("MEM")
        .map_err(|e| DataFusionError::Execution(format!("Failed to get MEM driver: {}", e)))?;

    let mut out_dataset = create_output_dataset(&mem_driver, out_width, out_height, &band_type)?;

    let geotransform = [
        out_ulx,
        ref_md.scale_x(),
        ref_md.skew_x(),
        out_uly,
        ref_md.skew_y(),
        ref_md.scale_y(),
    ];
    out_dataset
        .set_geo_transform(&geotransform)
        .map_err(|e| DataFusionError::Execution(format!("Failed to set geotransform: {}", e)))?;

    // Set spatial reference based on reference raster dataset (if present)
    let provider = crate::gdal_dataset_provider::thread_local_provider()
        .map_err(|e| DataFusionError::Execution(format!("Failed to init GDAL provider: {}", e)))?;
    let ref_raster_ds = provider
        .raster_ref_to_gdal(reference_raster)
        .map_err(|e| DataFusionError::Execution(format!("Failed to create GDAL dataset: {}", e)))?;
    if let Ok(srs) = ref_raster_ds.as_dataset().spatial_ref() {
        out_dataset.set_spatial_ref(&srs).map_err(|e| {
            DataFusionError::Execution(format!("Failed to set spatial reference: {}", e))
        })?;
    }

    // Initialize output band to nodata (if provided) or 0
    let init_value = nodata_value.unwrap_or(0.0);
    initialize_band(
        &mut out_dataset,
        &band_type,
        out_width,
        out_height,
        init_value,
    )?;

    // Set nodata metadata on band
    if let Some(nodata) = nodata_value {
        let mut band = out_dataset
            .rasterband(1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to get output band: {}", e)))?;
        match band_type {
            BandDataType::UInt64 => {
                band.set_no_data_value_u64(Some(nodata as u64))
                    .map_err(|e| {
                        DataFusionError::Execution(format!("Failed to set nodata value: {}", e))
                    })?;
            }
            BandDataType::Int64 => {
                band.set_no_data_value_i64(Some(nodata as i64))
                    .map_err(|e| {
                        DataFusionError::Execution(format!("Failed to set nodata value: {}", e))
                    })?;
            }
            _ => band.set_no_data_value(Some(nodata)).map_err(|e| {
                DataFusionError::Execution(format!("Failed to set nodata value: {}", e))
            })?,
        }
    }

    rasterize_affine(
        &mut out_dataset,
        &[1],
        &[geometry],
        &[burn_value],
        all_touched,
    )
    .map_err(|e| DataFusionError::Execution(format!("Failed to rasterize geometry: {}", e)))?;

    // Read band data as bytes
    let band_bytes = read_band_as_bytes(&out_dataset, 1, out_width, out_height, &band_type)?;

    let out_metadata = RasterMetadata {
        width: out_width as u64,
        height: out_height as u64,
        upperleft_x: out_ulx,
        upperleft_y: out_uly,
        scale_x: ref_md.scale_x(),
        scale_y: ref_md.scale_y(),
        skew_x: ref_md.skew_x(),
        skew_y: ref_md.skew_y(),
    };

    let out_band_metadata = BandMetadata {
        nodata_value: nodata_value.map(|v| nodata_f64_to_bytes(v, &band_type)),
        storage_type: StorageType::InDb,
        datatype: band_type,
        outdb_url: None,
        outdb_band_id: None,
    };

    Ok((out_metadata, out_band_metadata, band_bytes))
}

fn create_output_dataset(
    mem_driver: &gdal::Driver,
    width: usize,
    height: usize,
    band_type: &BandDataType,
) -> Result<gdal::Dataset> {
    match band_type {
        BandDataType::UInt8 => mem_driver
            .create_with_band_type::<u8, _>("", width, height, 1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create dataset: {}", e))),
        BandDataType::Int8 => mem_driver
            .create_with_band_type::<i8, _>("", width, height, 1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create dataset: {}", e))),
        BandDataType::UInt16 => mem_driver
            .create_with_band_type::<u16, _>("", width, height, 1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create dataset: {}", e))),
        BandDataType::Int16 => mem_driver
            .create_with_band_type::<i16, _>("", width, height, 1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create dataset: {}", e))),
        BandDataType::UInt32 => mem_driver
            .create_with_band_type::<u32, _>("", width, height, 1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create dataset: {}", e))),
        BandDataType::Int32 => mem_driver
            .create_with_band_type::<i32, _>("", width, height, 1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create dataset: {}", e))),
        BandDataType::UInt64 => mem_driver
            .create_with_band_type::<u64, _>("", width, height, 1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create dataset: {}", e))),
        BandDataType::Int64 => mem_driver
            .create_with_band_type::<i64, _>("", width, height, 1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create dataset: {}", e))),
        BandDataType::Float32 => mem_driver
            .create_with_band_type::<f32, _>("", width, height, 1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create dataset: {}", e))),
        BandDataType::Float64 => mem_driver
            .create_with_band_type::<f64, _>("", width, height, 1)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create dataset: {}", e))),
    }
}

fn initialize_band(
    dataset: &mut gdal::Dataset,
    band_type: &BandDataType,
    width: usize,
    height: usize,
    init_value: f64,
) -> Result<()> {
    match band_type {
        BandDataType::UInt8 => initialize_band_t::<u8>(dataset, width, height, init_value as u8),
        BandDataType::Int8 => initialize_band_t::<i8>(dataset, width, height, init_value as i8),
        BandDataType::UInt16 => initialize_band_t::<u16>(dataset, width, height, init_value as u16),
        BandDataType::Int16 => initialize_band_t::<i16>(dataset, width, height, init_value as i16),
        BandDataType::UInt32 => initialize_band_t::<u32>(dataset, width, height, init_value as u32),
        BandDataType::Int32 => initialize_band_t::<i32>(dataset, width, height, init_value as i32),
        BandDataType::UInt64 => initialize_band_t::<u64>(dataset, width, height, init_value as u64),
        BandDataType::Int64 => initialize_band_t::<i64>(dataset, width, height, init_value as i64),
        BandDataType::Float32 => {
            initialize_band_t::<f32>(dataset, width, height, init_value as f32)
        }
        BandDataType::Float64 => initialize_band_t::<f64>(dataset, width, height, init_value),
    }
}

fn initialize_band_t<T: gdal::raster::GdalType + Copy>(
    dataset: &mut gdal::Dataset,
    width: usize,
    height: usize,
    init_value: T,
) -> Result<()> {
    let mut band = dataset
        .rasterband(1)
        .map_err(|e| DataFusionError::Execution(format!("Failed to get output band: {}", e)))?;

    let values = vec![init_value; width * height];
    let mut buffer = Buffer::new((width, height), values);
    band.write((0, 0), (width, height), &mut buffer)
        .map_err(|e| DataFusionError::Execution(format!("Failed to initialize band: {}", e)))?;

    Ok(())
}

fn read_band_as_bytes(
    dataset: &gdal::Dataset,
    band_idx: usize,
    width: usize,
    height: usize,
    band_type: &BandDataType,
) -> Result<Vec<u8>> {
    let band = dataset.rasterband(band_idx).map_err(|e| {
        DataFusionError::Execution(format!("Failed to get band {}: {}", band_idx, e))
    })?;

    let data = match band_type {
        BandDataType::UInt8 => {
            let buffer = band
                .read_as::<u8>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to read band {} data: {}",
                        band_idx, e
                    ))
                })?;
            buffer.data().to_vec()
        }
        BandDataType::Int8 => {
            let buffer = band
                .read_as::<i8>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to read band {} data: {}",
                        band_idx, e
                    ))
                })?;
            buffer.data().iter().map(|v| *v as u8).collect()
        }
        BandDataType::UInt16 => {
            let buffer = band
                .read_as::<u16>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to read band {} data: {}",
                        band_idx, e
                    ))
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::Int16 => {
            let buffer = band
                .read_as::<i16>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to read band {} data: {}",
                        band_idx, e
                    ))
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::UInt32 => {
            let buffer = band
                .read_as::<u32>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to read band {} data: {}",
                        band_idx, e
                    ))
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::Int32 => {
            let buffer = band
                .read_as::<i32>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to read band {} data: {}",
                        band_idx, e
                    ))
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::UInt64 => {
            let buffer = band
                .read_as::<u64>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to read band {} data: {}",
                        band_idx, e
                    ))
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::Int64 => {
            let buffer = band
                .read_as::<i64>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to read band {} data: {}",
                        band_idx, e
                    ))
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::Float32 => {
            let buffer = band
                .read_as::<f32>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to read band {} data: {}",
                        band_idx, e
                    ))
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::Float64 => {
            let buffer = band
                .read_as::<f64>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to read band {} data: {}",
                        band_idx, e
                    ))
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
    };

    Ok(data)
}

// -----------------------------------------------------------------------------
// ColumnarValue helpers (copied from other UDF modules for consistency)
// -----------------------------------------------------------------------------

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

fn extract_bool_scalar(arg: &ColumnarValue) -> Result<Option<bool>> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Boolean(v)) => Ok(*v),
        _ => Ok(None),
    }
}

fn extract_f64_scalar(arg: &ColumnarValue) -> Result<Option<f64>> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Float64(v)) => Ok(*v),
        ColumnarValue::Scalar(ScalarValue::Float32(v)) => Ok(v.map(|x| x as f64)),
        ColumnarValue::Scalar(ScalarValue::Int64(v)) => Ok(v.map(|x| x as f64)),
        ColumnarValue::Scalar(ScalarValue::Int32(v)) => Ok(v.map(|x| x as f64)),
        ColumnarValue::Scalar(ScalarValue::Int16(v)) => Ok(v.map(|x| x as f64)),
        ColumnarValue::Scalar(ScalarValue::Int8(v)) => Ok(v.map(|x| x as f64)),
        ColumnarValue::Scalar(ScalarValue::UInt64(v)) => Ok(v.map(|x| x as f64)),
        ColumnarValue::Scalar(ScalarValue::UInt32(v)) => Ok(v.map(|x| x as f64)),
        ColumnarValue::Scalar(ScalarValue::UInt16(v)) => Ok(v.map(|x| x as f64)),
        ColumnarValue::Scalar(ScalarValue::UInt8(v)) => Ok(v.map(|x| x as f64)),
        _ => Ok(None),
    }
}

fn extract_string_scalar(arg: &ColumnarValue) -> Result<Option<String>> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Utf8(v)) => Ok(v.clone()),
        ColumnarValue::Scalar(ScalarValue::LargeUtf8(v)) => Ok(v.clone()),
        _ => Ok(None),
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

fn finish_result(args: &[ColumnarValue], out: ArrayRef) -> Result<ColumnarValue> {
    for arg in args {
        if let ColumnarValue::Array(_) = arg {
            return Ok(ColumnarValue::Array(out));
        }
    }
    Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(&out, 0)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::rs_from_gdal_raster::RsFromGDALRaster;
    use gdal::vector::Geometry;
    use sedona_raster::array::RasterStructArray;

    fn bytes_to_f64_vec(bytes: &[u8]) -> Vec<f64> {
        bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn test_parse_pixel_type() {
        assert_eq!(parse_pixel_type("D").unwrap(), BandDataType::Float64);
        assert_eq!(parse_pixel_type("f").unwrap(), BandDataType::Float32);
        assert_eq!(parse_pixel_type("I").unwrap(), BandDataType::Int32);
        assert_eq!(parse_pixel_type("S").unwrap(), BandDataType::Int16);
        assert_eq!(parse_pixel_type("US").unwrap(), BandDataType::UInt16);
        assert_eq!(parse_pixel_type("B").unwrap(), BandDataType::UInt8);
        assert_eq!(parse_pixel_type("I8").unwrap(), BandDataType::Int8);
        assert_eq!(parse_pixel_type("U64").unwrap(), BandDataType::UInt64);
        assert_eq!(parse_pixel_type("I64").unwrap(), BandDataType::Int64);
    }

    #[test]
    fn test_rs_as_raster_use_reference_extent() {
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let md = raster.metadata();

        // Build a polygon matching the first pixel footprint.
        let ulx = md.upper_left_x();
        let uly = md.upper_left_y();
        let scale_x = md.scale_x();
        let scale_y = md.scale_y();

        let minx = ulx;
        let maxx = ulx + scale_x;
        let maxy = uly;
        let miny = uly + scale_y; // scale_y is typically negative

        let wkt = format!(
            "POLYGON(({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))"
        );
        let geom = Geometry::from_wkt(&wkt).unwrap();
        let geom_wkb = geom.iso_wkb().unwrap();

        let (out_md, _band_md, out_bytes) = as_raster(
            &geom_wkb,
            &raster,
            BandDataType::Float64,
            false,
            255.0,
            Some(0.0),
            false,
        )
        .unwrap();

        assert_eq!(out_md.width, md.width() as u64);
        assert_eq!(out_md.height, md.height() as u64);
        assert_eq!(out_md.upperleft_x, md.upper_left_x());
        assert_eq!(out_md.upperleft_y, md.upper_left_y());

        // Ensure the first pixel was burned.
        let values = bytes_to_f64_vec(&out_bytes);
        assert_eq!(values[0], 255.0);
    }

    #[test]
    fn test_rs_as_raster_use_geometry_extent() {
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let md = raster.metadata();

        // Same single-pixel polygon as above.
        let ulx = md.upper_left_x();
        let uly = md.upper_left_y();
        let scale_x = md.scale_x();
        let scale_y = md.scale_y();

        let minx = ulx;
        let maxx = ulx + scale_x;
        let maxy = uly;
        let miny = uly + scale_y;

        let wkt = format!(
            "POLYGON(({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))"
        );
        let geom = Geometry::from_wkt(&wkt).unwrap();
        let geom_wkb = geom.iso_wkb().unwrap();

        let (out_md, _band_md, out_bytes) = as_raster(
            &geom_wkb,
            &raster,
            BandDataType::Float64,
            false,
            255.0,
            Some(0.0),
            true,
        )
        .unwrap();

        assert_eq!(out_md.width, 1);
        assert_eq!(out_md.height, 1);
        assert_eq!(out_md.upperleft_x, md.upper_left_x());
        assert_eq!(out_md.upperleft_y, md.upper_left_y());

        let values = bytes_to_f64_vec(&out_bytes);
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], 255.0);
    }
}
