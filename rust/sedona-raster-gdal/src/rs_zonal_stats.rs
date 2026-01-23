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

//! RS_ZonalStats and RS_ZonalStatsAll UDFs - Compute statistics for pixels within a geometry
//!
//! RS_ZonalStats computes a single statistic (count, sum, mean, median, mode, stddev, variance, min, max)
//! for all pixels within a geometry boundary.
//!
//! RS_ZonalStatsAll computes all statistics and returns them as a struct.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, Int64Builder, StructBuilder};
use arrow_array::{Array, ArrayRef, BinaryArray, StructArray};
use arrow_schema::{DataType, Field, Fields};
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
use sedona_raster::traits::RasterRef;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::BandDataType;

use crate::dataset::{nodata_bytes_to_f64, raster_to_dataset};

/// Statistics types supported by RS_ZonalStats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatType {
    Count,
    Sum,
    Mean,
    Median,
    Mode,
    StdDev,
    Variance,
    Min,
    Max,
}

impl StatType {
    /// Parse stat type from string (case-insensitive)
    fn from_str(s: &str) -> Option<StatType> {
        match s.to_lowercase().as_str() {
            "count" => Some(StatType::Count),
            "sum" => Some(StatType::Sum),
            "mean" | "avg" | "average" => Some(StatType::Mean),
            "median" => Some(StatType::Median),
            "mode" => Some(StatType::Mode),
            "stddev" | "std" | "standarddeviation" => Some(StatType::StdDev),
            "variance" | "var" => Some(StatType::Variance),
            "min" | "minimum" => Some(StatType::Min),
            "max" | "maximum" => Some(StatType::Max),
            _ => None,
        }
    }
}

/// Computed statistics for a zone
#[derive(Debug, Default)]
pub struct ZonalStatistics {
    pub count: i64,
    pub sum: f64,
    pub mean: f64,
    pub median: f64,
    pub mode: f64,
    pub stddev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
}

impl ZonalStatistics {
    /// Get a specific statistic value
    fn get(&self, stat_type: StatType) -> f64 {
        match stat_type {
            StatType::Count => self.count as f64,
            StatType::Sum => self.sum,
            StatType::Mean => self.mean,
            StatType::Median => self.median,
            StatType::Mode => self.mode,
            StatType::StdDev => self.stddev,
            StatType::Variance => self.variance,
            StatType::Min => self.min,
            StatType::Max => self.max,
        }
    }
}

// =============================================================================
// RS_ZonalStats UDF
// =============================================================================

/// RS_ZonalStats() scalar UDF implementation
///
/// Computes a single statistic for pixels within a geometry
pub fn rs_zonal_stats_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_zonalstats",
        vec![
            Arc::new(RsZonalStats {
                with_band: false,
                with_options: false,
            }),
            Arc::new(RsZonalStats {
                with_band: true,
                with_options: false,
            }),
            Arc::new(RsZonalStats {
                with_band: true,
                with_options: true,
            }),
        ],
        Volatility::Immutable,
        Some(rs_zonal_stats_doc()),
    )
}

fn rs_zonal_stats_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Computes a statistic (count, sum, mean, median, mode, stddev, variance, min, max) for pixels within a geometry.".to_string(),
        "RS_ZonalStats(raster: Raster, geometry: Geometry, statType: String) or RS_ZonalStats(raster: Raster, band: Integer, geometry: Geometry, statType: String, allTouched: Boolean, excludeNoData: Boolean)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("geometry", "Geometry: Zone geometry (WKB format)")
    .with_argument("statType", "String: Statistic type (count, sum, mean, median, mode, stddev, variance, min, max)")
    .with_argument("band", "Integer: Band number (1-based, default: 1)")
    .with_argument("allTouched", "Boolean: Include all touched pixels (default: false)")
    .with_argument("excludeNoData", "Boolean: Exclude nodata values (default: true)")
    .with_sql_example("SELECT RS_ZonalStats(raster, ST_GeomFromText('POLYGON((...))'), 'mean') FROM raster_table".to_string())
    .build()
}

/// Kernel implementation for RS_ZonalStats
#[derive(Debug)]
struct RsZonalStats {
    with_band: bool,
    with_options: bool,
}

impl SedonaScalarKernel for RsZonalStats {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = if self.with_options {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_string(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_boolean(),
            ]
        } else if self.with_band {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_string(),
            ]
        } else {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_string(),
            ]
        };

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(DataType::Float64));
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let num_iterations = calc_num_iterations(args);

        // Parse arguments
        let (geom_arg_idx, stat_arg_idx, band_num, all_touched, exclude_nodata) =
            if self.with_options {
                let band = extract_i32_scalar(&args[1])?.unwrap_or(1) as usize;
                let all_touched = extract_bool_scalar(&args[4])?.unwrap_or(false);
                let exclude_nodata = extract_bool_scalar(&args[5])?.unwrap_or(true);
                (2, 3, band, all_touched, exclude_nodata)
            } else if self.with_band {
                let band = extract_i32_scalar(&args[1])?.unwrap_or(1) as usize;
                (2, 3, band, false, true)
            } else {
                (1, 2, 1, false, true)
            };

        // Get stat type
        let stat_str = extract_string_scalar(&args[stat_arg_idx])?
            .ok_or_else(|| DataFusionError::Execution("Stat type is required".to_string()))?;
        let stat_type = StatType::from_str(&stat_str).ok_or_else(|| {
            DataFusionError::Execution(format!("Unknown stat type: {}", stat_str))
        })?;

        // Get raster and geometry arrays
        let raster_array = get_raster_array(&args[0])?;
        let geom_array = get_binary_array(&args[geom_arg_idx])?;

        // Build results
        let mut builder = Float64Builder::with_capacity(num_iterations);

        for i in 0..num_iterations {
            let raster_idx = if raster_array.len() == 1 { 0 } else { i };
            let geom_idx = if geom_array.len() == 1 { 0 } else { i };

            if raster_array.is_null(raster_idx) || geom_array.is_null(geom_idx) {
                builder.append_null();
                continue;
            }

            let raster = raster_array.get(raster_idx)?;
            let geom_wkb = geom_array.value(geom_idx);

            match compute_zonal_stats(&raster, geom_wkb, band_num, all_touched, exclude_nodata) {
                Ok(stats) => {
                    builder.append_value(stats.get(stat_type));
                }
                Err(e) => {
                    eprintln!("RS_ZonalStats error: {}", e);
                    builder.append_null();
                }
            }
        }

        finish_result(args, Arc::new(builder.finish()))
    }
}

// =============================================================================
// RS_ZonalStatsAll UDF
// =============================================================================

/// RS_ZonalStatsAll() scalar UDF implementation
///
/// Computes all statistics for pixels within a geometry and returns a struct
pub fn rs_zonal_stats_all_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_zonalstatsall",
        vec![
            Arc::new(RsZonalStatsAll {
                with_band: false,
                with_options: false,
            }),
            Arc::new(RsZonalStatsAll {
                with_band: true,
                with_options: false,
            }),
            Arc::new(RsZonalStatsAll {
                with_band: true,
                with_options: true,
            }),
        ],
        Volatility::Immutable,
        Some(rs_zonal_stats_all_doc()),
    )
}

fn rs_zonal_stats_all_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Computes all statistics (count, sum, mean, median, mode, stddev, variance, min, max) for pixels within a geometry and returns them as a struct.".to_string(),
        "RS_ZonalStatsAll(raster: Raster, geometry: Geometry) or RS_ZonalStatsAll(raster: Raster, band: Integer, geometry: Geometry, allTouched: Boolean, excludeNoData: Boolean)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("geometry", "Geometry: Zone geometry (WKB format)")
    .with_argument("band", "Integer: Band number (1-based, default: 1)")
    .with_argument("allTouched", "Boolean: Include all touched pixels (default: false)")
    .with_argument("excludeNoData", "Boolean: Exclude nodata values (default: true)")
    .with_sql_example("SELECT RS_ZonalStatsAll(raster, ST_GeomFromText('POLYGON((...))')) FROM raster_table".to_string())
    .build()
}

/// Kernel implementation for RS_ZonalStatsAll
#[derive(Debug)]
struct RsZonalStatsAll {
    with_band: bool,
    with_options: bool,
}

impl SedonaScalarKernel for RsZonalStatsAll {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = if self.with_options {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_boolean(),
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

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(zonal_stats_struct_type()));
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let num_iterations = calc_num_iterations(args);

        // Parse arguments
        let (geom_arg_idx, band_num, all_touched, exclude_nodata) = if self.with_options {
            let band = extract_i32_scalar(&args[1])?.unwrap_or(1) as usize;
            let all_touched = extract_bool_scalar(&args[3])?.unwrap_or(false);
            let exclude_nodata = extract_bool_scalar(&args[4])?.unwrap_or(true);
            (2, band, all_touched, exclude_nodata)
        } else if self.with_band {
            let band = extract_i32_scalar(&args[1])?.unwrap_or(1) as usize;
            (2, band, false, true)
        } else {
            (1, 1, false, true)
        };

        // Get raster and geometry arrays
        let raster_array = get_raster_array(&args[0])?;
        let geom_array = get_binary_array(&args[geom_arg_idx])?;

        // Build struct result
        let fields = zonal_stats_struct_fields();
        let mut builder = StructBuilder::from_fields(fields, num_iterations);

        for i in 0..num_iterations {
            let raster_idx = if raster_array.len() == 1 { 0 } else { i };
            let geom_idx = if geom_array.len() == 1 { 0 } else { i };

            if raster_array.is_null(raster_idx) || geom_array.is_null(geom_idx) {
                // Append nulls for all fields
                builder
                    .field_builder::<Int64Builder>(0)
                    .unwrap()
                    .append_null();
                for j in 1..9 {
                    builder
                        .field_builder::<Float64Builder>(j)
                        .unwrap()
                        .append_null();
                }
                builder.append_null();
                continue;
            }

            let raster = raster_array.get(raster_idx)?;
            let geom_wkb = geom_array.value(geom_idx);

            match compute_zonal_stats(&raster, geom_wkb, band_num, all_touched, exclude_nodata) {
                Ok(stats) => {
                    builder
                        .field_builder::<Int64Builder>(0)
                        .unwrap()
                        .append_value(stats.count);
                    builder
                        .field_builder::<Float64Builder>(1)
                        .unwrap()
                        .append_value(stats.sum);
                    builder
                        .field_builder::<Float64Builder>(2)
                        .unwrap()
                        .append_value(stats.mean);
                    builder
                        .field_builder::<Float64Builder>(3)
                        .unwrap()
                        .append_value(stats.median);
                    builder
                        .field_builder::<Float64Builder>(4)
                        .unwrap()
                        .append_value(stats.mode);
                    builder
                        .field_builder::<Float64Builder>(5)
                        .unwrap()
                        .append_value(stats.stddev);
                    builder
                        .field_builder::<Float64Builder>(6)
                        .unwrap()
                        .append_value(stats.variance);
                    builder
                        .field_builder::<Float64Builder>(7)
                        .unwrap()
                        .append_value(stats.min);
                    builder
                        .field_builder::<Float64Builder>(8)
                        .unwrap()
                        .append_value(stats.max);
                    builder.append(true);
                }
                Err(e) => {
                    eprintln!("RS_ZonalStatsAll error: {}", e);
                    builder
                        .field_builder::<Int64Builder>(0)
                        .unwrap()
                        .append_null();
                    for j in 1..9 {
                        builder
                            .field_builder::<Float64Builder>(j)
                            .unwrap()
                            .append_null();
                    }
                    builder.append_null();
                }
            }
        }

        let result = Arc::new(builder.finish()) as ArrayRef;
        finish_result(args, result)
    }
}

/// Return type for ZonalStatsAll struct
fn zonal_stats_struct_type() -> DataType {
    DataType::Struct(zonal_stats_struct_fields())
}

/// Fields for the ZonalStatsAll struct
fn zonal_stats_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("count", DataType::Int64, true),
        Field::new("sum", DataType::Float64, true),
        Field::new("mean", DataType::Float64, true),
        Field::new("median", DataType::Float64, true),
        Field::new("mode", DataType::Float64, true),
        Field::new("stddev", DataType::Float64, true),
        Field::new("variance", DataType::Float64, true),
        Field::new("min", DataType::Float64, true),
        Field::new("max", DataType::Float64, true),
    ])
}

// =============================================================================
// Core Statistics Computation
// =============================================================================

/// Compute zonal statistics for a raster within a geometry
fn compute_zonal_stats(
    raster: &RasterRefImpl<'_>,
    geom_wkb: &[u8],
    band_num: usize,
    all_touched: bool,
    exclude_nodata: bool,
) -> Result<ZonalStatistics> {
    let metadata = raster.metadata();
    let bands = raster.bands();
    let width = metadata.width() as usize;
    let height = metadata.height() as usize;

    // Validate band number
    if band_num == 0 || band_num > bands.len() {
        return Err(DataFusionError::Execution(format!(
            "Band {} is out of range (1-{})",
            band_num,
            bands.len()
        )));
    }

    // Parse geometry from WKB
    let geometry = Geometry::from_wkb(geom_wkb).map_err(|e| {
        DataFusionError::Execution(format!("Failed to parse geometry from WKB: {}", e))
    })?;

    // Create GDAL dataset from raster
    let gdal_dataset = raster_to_dataset(raster)
        .map_err(|e| DataFusionError::Execution(format!("Failed to create GDAL dataset: {}", e)))?;

    // Create a mask raster
    let mem_driver = DriverManager::get_driver_by_name("MEM")
        .map_err(|e| DataFusionError::Execution(format!("Failed to get MEM driver: {}", e)))?;

    let mut mask_dataset = mem_driver
        .create_with_band_type::<u8, _>("", width, height, 1)
        .map_err(|e| DataFusionError::Execution(format!("Failed to create mask dataset: {}", e)))?;

    // Set geotransform
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

    // Set spatial reference
    if let Ok(srs) = gdal_dataset.spatial_ref() {
        mask_dataset.set_spatial_ref(&srs).map_err(|e| {
            DataFusionError::Execution(format!("Failed to set spatial reference: {}", e))
        })?;
    }

    // Initialize mask to 0
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

    // Rasterize geometry
    let rasterize_options = RasterizeOptions {
        all_touched,
        ..Default::default()
    };

    rasterize(
        &mut mask_dataset,
        &[1],
        &[geometry],
        &[1.0],
        Some(rasterize_options),
    )
    .map_err(|e| DataFusionError::Execution(format!("Failed to rasterize geometry: {}", e)))?;

    // Read mask
    let mask_band = mask_dataset
        .rasterband(1)
        .map_err(|e| DataFusionError::Execution(format!("Failed to get mask band: {}", e)))?;
    let mask_buffer = mask_band
        .read_as::<u8>((0, 0), (width, height), (width, height), None)
        .map_err(|e| DataFusionError::Execution(format!("Failed to read mask: {}", e)))?;
    let mask = mask_buffer.data();

    // Get band data
    let band = bands
        .band(band_num)
        .map_err(|e| DataFusionError::Execution(format!("Failed to get band: {}", e)))?;
    let band_metadata = band.metadata();
    let data_type = band_metadata.data_type();
    let band_data = band.data();

    // Get nodata value
    let nodata = nodata_bytes_to_f64(band_metadata.nodata_value(), &data_type);

    // Collect pixel values within the geometry
    let mut values: Vec<f64> = Vec::new();

    for (pixel_idx, &mask_val) in mask.iter().enumerate().take(width * height) {
        if mask_val == 1 {
            let value = read_pixel_value(band_data, pixel_idx, &data_type)?;

            // Check for nodata
            if exclude_nodata {
                if let Some(no_data) = nodata {
                    if (value - no_data).abs() < f64::EPSILON || value.is_nan() {
                        continue;
                    }
                }
            }

            values.push(value);
        }
    }

    // Compute statistics
    compute_statistics(&values)
}

/// Compute all statistics from a vector of values
fn compute_statistics(values: &[f64]) -> Result<ZonalStatistics> {
    if values.is_empty() {
        return Ok(ZonalStatistics {
            count: 0,
            sum: 0.0,
            mean: f64::NAN,
            median: f64::NAN,
            mode: f64::NAN,
            stddev: f64::NAN,
            variance: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
        });
    }

    let count = values.len() as i64;
    let sum: f64 = values.iter().sum();
    let mean = sum / count as f64;

    // Min and max
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Variance and standard deviation
    let variance = if count > 1 {
        let sum_sq_diff: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum();
        sum_sq_diff / (count as f64 - 1.0) // Sample variance
    } else {
        0.0
    };
    let stddev = variance.sqrt();

    // Median
    let median = {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };

    // Mode (most frequent value)
    let mode = {
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &v in values {
            // Quantize to avoid floating point comparison issues
            let key = (v * 1_000_000.0).round() as i64;
            *counts.entry(key).or_insert(0) += 1;
        }
        let (mode_key, _) = counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .unwrap_or((0, 0));
        mode_key as f64 / 1_000_000.0
    };

    Ok(ZonalStatistics {
        count,
        sum,
        mean,
        median,
        mode,
        stddev,
        variance,
        min,
        max,
    })
}

/// Read pixel value from band data
fn read_pixel_value(data: &[u8], offset: usize, data_type: &BandDataType) -> Result<f64> {
    let byte_size = data_type_byte_size(data_type);
    let byte_offset = offset * byte_size;

    if byte_offset + byte_size > data.len() {
        return Err(DataFusionError::Execution(
            "Pixel offset out of bounds".to_string(),
        ));
    }

    let value = match data_type {
        BandDataType::UInt8 => data[byte_offset] as f64,
        BandDataType::UInt16 => {
            u16::from_le_bytes([data[byte_offset], data[byte_offset + 1]]) as f64
        }
        BandDataType::Int16 => {
            i16::from_le_bytes([data[byte_offset], data[byte_offset + 1]]) as f64
        }
        BandDataType::UInt32 => u32::from_le_bytes([
            data[byte_offset],
            data[byte_offset + 1],
            data[byte_offset + 2],
            data[byte_offset + 3],
        ]) as f64,
        BandDataType::Int32 => i32::from_le_bytes([
            data[byte_offset],
            data[byte_offset + 1],
            data[byte_offset + 2],
            data[byte_offset + 3],
        ]) as f64,
        BandDataType::Float32 => f32::from_le_bytes([
            data[byte_offset],
            data[byte_offset + 1],
            data[byte_offset + 2],
            data[byte_offset + 3],
        ]) as f64,
        BandDataType::Float64 => f64::from_le_bytes([
            data[byte_offset],
            data[byte_offset + 1],
            data[byte_offset + 2],
            data[byte_offset + 3],
            data[byte_offset + 4],
            data[byte_offset + 5],
            data[byte_offset + 6],
            data[byte_offset + 7],
        ]),
    };

    Ok(value)
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

// =============================================================================
// Helper Functions
// =============================================================================

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

/// Helper to extract bool scalar value
fn extract_bool_scalar(arg: &ColumnarValue) -> Result<Option<bool>> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Boolean(v)) => Ok(*v),
        _ => Ok(None),
    }
}

/// Helper to extract string scalar value
fn extract_string_scalar(arg: &ColumnarValue) -> Result<Option<String>> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Utf8(v)) => Ok(v.clone()),
        ColumnarValue::Scalar(ScalarValue::LargeUtf8(v)) => Ok(v.clone()),
        ColumnarValue::Scalar(ScalarValue::Utf8View(v)) => Ok(v.clone()),
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
    fn test_stat_type_from_str() {
        assert_eq!(StatType::from_str("count"), Some(StatType::Count));
        assert_eq!(StatType::from_str("COUNT"), Some(StatType::Count));
        assert_eq!(StatType::from_str("mean"), Some(StatType::Mean));
        assert_eq!(StatType::from_str("avg"), Some(StatType::Mean));
        assert_eq!(StatType::from_str("stddev"), Some(StatType::StdDev));
        assert_eq!(StatType::from_str("invalid"), None);
    }

    #[test]
    fn test_compute_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_statistics(&values).unwrap();

        assert_eq!(stats.count, 5);
        assert!((stats.sum - 15.0).abs() < f64::EPSILON);
        assert!((stats.mean - 3.0).abs() < f64::EPSILON);
        assert!((stats.median - 3.0).abs() < f64::EPSILON);
        assert!((stats.min - 1.0).abs() < f64::EPSILON);
        assert!((stats.max - 5.0).abs() < f64::EPSILON);
        // Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 4 = 10/4 = 2.5
        assert!((stats.variance - 2.5).abs() < 0.001);
        assert!((stats.stddev - 2.5_f64.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_compute_statistics_empty() {
        let values: Vec<f64> = vec![];
        let stats = compute_statistics(&values).unwrap();

        assert_eq!(stats.count, 0);
        assert_eq!(stats.sum, 0.0);
        assert!(stats.mean.is_nan());
        assert!(stats.min.is_nan());
        assert!(stats.max.is_nan());
    }

    #[test]
    fn test_compute_statistics_single() {
        let values = vec![42.0];
        let stats = compute_statistics(&values).unwrap();

        assert_eq!(stats.count, 1);
        assert!((stats.sum - 42.0).abs() < f64::EPSILON);
        assert!((stats.mean - 42.0).abs() < f64::EPSILON);
        assert!((stats.median - 42.0).abs() < f64::EPSILON);
        assert!((stats.min - 42.0).abs() < f64::EPSILON);
        assert!((stats.max - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rs_zonal_stats_with_test_raster() {
        use crate::rs_from_gdal_raster::RsFromGDALRaster;
        use gdal::vector::Geometry;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();

        // Create a polygon covering the entire raster
        let metadata = raster.metadata();
        let min_x = metadata.upper_left_x();
        let max_y = metadata.upper_left_y();
        let max_x = min_x + (metadata.width() as f64 * metadata.scale_x());
        let min_y = max_y + (metadata.height() as f64 * metadata.scale_y());

        let wkt = format!(
            "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
            min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
        );

        let geometry = Geometry::from_wkt(&wkt).unwrap();
        let geom_wkb = geometry.wkb().unwrap();

        let result = compute_zonal_stats(&raster, &geom_wkb, 1, false, true);
        assert!(
            result.is_ok(),
            "Zonal stats should succeed: {:?}",
            result.err()
        );

        let stats = result.unwrap();
        assert!(stats.count > 0, "Should have some pixels");
        assert!(stats.min <= stats.max, "Min should be <= max");
        assert!(
            stats.min <= stats.mean && stats.mean <= stats.max,
            "Mean should be between min and max"
        );
    }
}
