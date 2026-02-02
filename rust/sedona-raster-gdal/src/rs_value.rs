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

//! RS_Value UDF - Get raster pixel value at a point or grid coordinate
//!
//! Returns the value at the given point in the raster. If no band number is specified,
//! it defaults to 1. If the CRS of the input point differs from the raster CRS,
//! the point will be transformed to match the raster CRS.

use std::sync::Arc;

use arrow_array::builder::Float64Builder;
use arrow_array::{ArrayRef, Int32Array, StructArray};
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};

use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::affine_transformation::to_raster_coordinate;
use sedona_raster::array::{RasterRefImpl, RasterStructArray};
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::BandDataType;

use crate::crs_utils;
use crate::raster_band_reader::RasterBandReader;

/// RS_Value() scalar UDF implementation
///
/// Returns the value at the given point in the raster
pub fn rs_value_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_value",
        vec![
            Arc::new(RsValuePoint { with_band: false }),
            Arc::new(RsValuePoint { with_band: true }),
            Arc::new(RsValueGrid),
        ],
        Volatility::Immutable,
        Some(rs_value_doc()),
    )
}

fn rs_value_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the value at the given point in the raster. If no band number is specified it defaults to 1.".to_string(),
        "RS_Value(raster: Raster, point: Geometry) or RS_Value(raster: Raster, point: Geometry, band: Integer) or RS_Value(raster: Raster, colX: Integer, rowY: Integer, band: Integer)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("point", "Geometry: Point to sample (CRS will be transformed if needed)")
    .with_argument("colX", "Integer: Column X coordinate (0-based)")
    .with_argument("rowY", "Integer: Row Y coordinate (0-based)")
    .with_argument("band", "Integer: Band number (1-based, defaults to 1)")
    .with_sql_example("SELECT RS_Value(raster, ST_Point(-13077301.685, 4002565.802)) FROM raster_table".to_string())
    .build()
}

/// Kernel for RS_Value with point geometry argument
#[derive(Debug)]
struct RsValuePoint {
    with_band: bool,
}

impl SedonaScalarKernel for RsValuePoint {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = if self.with_band {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_integer(),
            ]
        } else {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
            ]
        };

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(DataType::Float64));
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let num_iterations = calc_num_iterations(args);
        let mut builder = Float64Builder::with_capacity(num_iterations);

        let band_array = if self.with_band {
            args[2]
                .clone()
                .cast_to(&DataType::Int32, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Int32(Some(1)).to_array_of_size(num_iterations)?
        };
        let band_array = band_array
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| DataFusionError::Internal("Expected Int32Array for band".to_string()))?
            .clone();
        let mut band_iter = band_array.iter();

        let exec_arg_types = vec![arg_types[0].clone(), arg_types[1].clone()];
        let exec_args = vec![args[0].clone(), args[1].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, maybe_point_crs| {
            let band_num = band_iter.next().flatten().unwrap_or(1) as usize;
            let (raster, point_wkb) = match (raster_opt, wkb_opt) {
                (Some(raster), Some(point_wkb)) => (raster, point_wkb),
                _ => {
                    builder.append_null();
                    return Ok(());
                }
            };

            let raster_crs = raster.crs();
            let point_wkb = if crs_utils::crs_equivalent(raster_crs, maybe_point_crs)? {
                point_wkb.to_vec()
            } else {
                crs_utils::transform_wkb_to_crs(point_wkb, maybe_point_crs, raster_crs)?
            };

            match get_value_at_point(raster, &point_wkb, band_num) {
                Ok(Some(value)) => builder.append_value(value),
                Ok(None) => builder.append_null(),
                Err(_) => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Kernel for RS_Value with grid coordinates
#[derive(Debug)]
struct RsValueGrid;

impl SedonaScalarKernel for RsValueGrid {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_integer(),
            ],
            SedonaType::Arrow(DataType::Float64),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let num_iterations = calc_num_iterations(args);
        let mut builder = Float64Builder::with_capacity(num_iterations);

        // Get scalar values for col_x, row_y, band
        let col_x = extract_i32_scalar(&args[1])?;
        let row_y = extract_i32_scalar(&args[2])?;
        let band_num = extract_i32_scalar(&args[3])?.unwrap_or(1) as usize;

        // Get raster array
        let raster_array = get_raster_array(&args[0])?;

        for i in 0..num_iterations {
            let raster_idx = if raster_array.len() == 1 { 0 } else { i };

            if raster_array.is_null(raster_idx) || col_x.is_none() || row_y.is_none() {
                builder.append_null();
                continue;
            }

            let raster = raster_array.get(raster_idx)?;
            let x = col_x.unwrap() as i64;
            let y = row_y.unwrap() as i64;

            match get_value_at_grid(&raster, x, y, band_num) {
                Ok(Some(value)) => builder.append_value(value),
                Ok(None) => builder.append_null(),
                Err(_) => builder.append_null(),
            }
        }

        finish_result(args, Arc::new(builder.finish()))
    }
}

/// Get pixel value at a point geometry
fn get_value_at_point(
    raster: &RasterRefImpl<'_>,
    point_wkb: &[u8],
    band_num: usize,
) -> Result<Option<f64>> {
    // Parse point from WKB
    let (x, y) = parse_point_from_wkb(point_wkb)?;

    // Convert world coordinates to raster coordinates
    let (col, row) = to_raster_coordinate(raster, x, y)
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert coordinates: {}", e)))?;

    get_value_at_grid(raster, col, row, band_num)
}

/// Get pixel value at grid coordinates
fn get_value_at_grid(
    raster: &RasterRefImpl<'_>,
    col: i64,
    row: i64,
    band_num: usize,
) -> Result<Option<f64>> {
    let metadata = raster.metadata();
    let width = metadata.width() as i64;
    let height = metadata.height() as i64;

    // Check bounds
    if col < 0 || col >= width || row < 0 || row >= height {
        return Ok(None);
    }

    let bands = raster.bands();
    if band_num == 0 || band_num > bands.len() {
        return Err(DataFusionError::Execution(format!(
            "Band {} is out of range (1-{})",
            band_num,
            bands.len()
        )));
    }

    let band = bands.band(band_num).map_err(|e| {
        DataFusionError::Execution(format!("Failed to get band {}: {}", band_num, e))
    })?;

    let band_metadata = band.metadata();
    let mut band_reader = RasterBandReader::new(raster);
    let value = band_reader.read_pixel_f64(band_num, col as usize, row as usize)?;

    // Check for nodata
    if let Some(nodata_bytes) = band_metadata.nodata_value() {
        let nodata = read_nodata_value(nodata_bytes, band_metadata.data_type())?;
        if (value - nodata).abs() < f64::EPSILON {
            return Ok(None);
        }
    }

    Ok(Some(value))
}

/// Parse point coordinates from WKB
fn parse_point_from_wkb(wkb: &[u8]) -> Result<(f64, f64)> {
    // WKB Point structure:
    // - 1 byte: byte order (01 = little endian, 00 = big endian)
    // - 4 bytes: geometry type (1 = Point)
    // - 8 bytes: X coordinate (f64)
    // - 8 bytes: Y coordinate (f64)

    if wkb.len() < 21 {
        return Err(DataFusionError::Execution(
            "Invalid WKB: too short for Point geometry".to_string(),
        ));
    }

    let byte_order = wkb[0];
    let geom_type = if byte_order == 0x01 {
        // Little endian
        u32::from_le_bytes([wkb[1], wkb[2], wkb[3], wkb[4]])
    } else {
        // Big endian
        u32::from_be_bytes([wkb[1], wkb[2], wkb[3], wkb[4]])
    };

    // Check geometry type (1 = Point, may have Z/M flags in higher bits)
    let base_type = geom_type & 0xFF;
    if base_type != 1 {
        return Err(DataFusionError::Execution(format!(
            "Expected Point geometry (type 1), got type {}",
            base_type
        )));
    }

    let (x, y) = if byte_order == 0x01 {
        // Little endian
        let x = f64::from_le_bytes([
            wkb[5], wkb[6], wkb[7], wkb[8], wkb[9], wkb[10], wkb[11], wkb[12],
        ]);
        let y = f64::from_le_bytes([
            wkb[13], wkb[14], wkb[15], wkb[16], wkb[17], wkb[18], wkb[19], wkb[20],
        ]);
        (x, y)
    } else {
        // Big endian
        let x = f64::from_be_bytes([
            wkb[5], wkb[6], wkb[7], wkb[8], wkb[9], wkb[10], wkb[11], wkb[12],
        ]);
        let y = f64::from_be_bytes([
            wkb[13], wkb[14], wkb[15], wkb[16], wkb[17], wkb[18], wkb[19], wkb[20],
        ]);
        (x, y)
    };

    Ok((x, y))
}

/// Read nodata value from bytes
fn read_nodata_value(bytes: &[u8], data_type: BandDataType) -> Result<f64> {
    match data_type {
        BandDataType::UInt8 => {
            if !bytes.is_empty() {
                Ok(bytes[0] as f64)
            } else {
                Err(DataFusionError::Execution(
                    "Invalid nodata bytes".to_string(),
                ))
            }
        }
        BandDataType::UInt16 => {
            if bytes.len() >= 2 {
                Ok(u16::from_le_bytes([bytes[0], bytes[1]]) as f64)
            } else {
                Err(DataFusionError::Execution(
                    "Invalid nodata bytes".to_string(),
                ))
            }
        }
        BandDataType::Int16 => {
            if bytes.len() >= 2 {
                Ok(i16::from_le_bytes([bytes[0], bytes[1]]) as f64)
            } else {
                Err(DataFusionError::Execution(
                    "Invalid nodata bytes".to_string(),
                ))
            }
        }
        BandDataType::UInt32 => {
            if bytes.len() >= 4 {
                Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
            } else {
                Err(DataFusionError::Execution(
                    "Invalid nodata bytes".to_string(),
                ))
            }
        }
        BandDataType::Int32 => {
            if bytes.len() >= 4 {
                Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
            } else {
                Err(DataFusionError::Execution(
                    "Invalid nodata bytes".to_string(),
                ))
            }
        }
        BandDataType::Float32 => {
            if bytes.len() >= 4 {
                Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
            } else {
                Err(DataFusionError::Execution(
                    "Invalid nodata bytes".to_string(),
                ))
            }
        }
        BandDataType::Float64 => {
            if bytes.len() >= 8 {
                Ok(f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]))
            } else {
                Err(DataFusionError::Execution(
                    "Invalid nodata bytes".to_string(),
                ))
            }
        }
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

/// Helper to extract i32 scalar value
fn extract_i32_scalar(arg: &ColumnarValue) -> Result<Option<i32>> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Int32(v)) => Ok(*v),
        ColumnarValue::Scalar(ScalarValue::Int64(v)) => Ok(v.map(|x| x as i32)),
        ColumnarValue::Scalar(ScalarValue::Int16(v)) => Ok(v.map(|x| x as i32)),
        ColumnarValue::Scalar(ScalarValue::Int8(v)) => Ok(v.map(|x| x as i32)),
        ColumnarValue::Array(array) => {
            if array.len() == 1 && !array.is_null(0) {
                if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
                    Ok(Some(arr.value(0)))
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        }
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
    use sedona_raster::affine_transformation::to_world_coordinate;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::{Edges, RASTER};
    use sedona_schema::raster::BandDataType;
    use sedona_testing::create::make_wkb;

    fn web_mercator_from_lonlat(lon: f64, lat: f64) -> (f64, f64) {
        let radius = 6378137.0_f64;
        let x = lon.to_radians() * radius;
        let y = (std::f64::consts::PI / 4.0 + lat.to_radians() / 2.0)
            .tan()
            .ln()
            * radius;
        (x, y)
    }

    #[test]
    fn test_parse_point_from_wkb() {
        // Little-endian WKB for POINT(1.0, 2.0)
        let wkb = [
            0x01, // Little endian
            0x01, 0x00, 0x00, 0x00, // Point type
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // X = 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // Y = 2.0
        ];

        let (x, y) = parse_point_from_wkb(&wkb).unwrap();
        assert!((x - 1.0).abs() < f64::EPSILON);
        assert!((y - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_pixel_value_uint8() {
        let data = vec![42u8, 100, 200];
        let raster_array = sedona_testing::rasters::raster_from_single_band(
            3,
            1,
            BandDataType::UInt8,
            &data,
            None,
        );
        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let mut reader = RasterBandReader::new(&raster);
        let value = reader.read_pixel_f64(1, 1, 0).unwrap();
        assert!((value - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_pixel_value_float32() {
        let mut data = vec![0u8; 12]; // 3 float32 values
        let values: [f32; 3] = [1.5, 2.5, 3.5];
        for (i, &v) in values.iter().enumerate() {
            data[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        let raster_array = sedona_testing::rasters::raster_from_single_band(
            3,
            1,
            BandDataType::Float32,
            &data,
            None,
        );
        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let mut reader = RasterBandReader::new(&raster);
        let value = reader.read_pixel_f64(1, 1, 0).unwrap();
        assert!((value - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rs_value_grid_with_test_raster() {
        // Load test raster and read value at grid coordinates
        use crate::rs_from_gdal_raster::RsFromGDALRaster;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        // Create a RasterStructArray to read values
        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();

        // Read pixel value at (0, 0) in band 1
        let value = get_value_at_grid(&raster, 0, 0, 1).unwrap();
        assert!(value.is_some());

        // Read pixel at center (5, 5) for a 10x10 raster
        let center_value = get_value_at_grid(&raster, 5, 5, 1).unwrap();
        assert!(center_value.is_some());

        // Read pixel outside bounds should return None
        let out_of_bounds = get_value_at_grid(&raster, 100, 100, 1).unwrap();
        assert!(out_of_bounds.is_none());
    }

    #[test]
    fn test_rs_value_invoke_grid() {
        // Test invoking RS_Value with grid coordinates
        use crate::rs_from_gdal_raster::RsFromGDALRaster;
        use arrow_schema::DataType;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_schema::datatypes::RASTER;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        let kernel = RsValueGrid;

        // Test return type
        let arg_types = vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ];
        let return_type = kernel.return_type(&arg_types).unwrap();
        assert!(return_type.is_some());

        // Test invoke_batch
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_array))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(0))), // col_x
            ColumnarValue::Scalar(ScalarValue::Int32(Some(0))), // row_y
            ColumnarValue::Scalar(ScalarValue::Int32(Some(1))), // band
        ];

        let result = kernel.invoke_batch(&arg_types, &args).unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => {
                // Value should be a valid pixel value
                assert!(value.is_finite());
            }
            _ => panic!("Expected Float64 scalar result"),
        }
    }

    #[test]
    fn test_rs_value_point_crs_transform() {
        use crate::rs_from_gdal_raster::RsFromGDALRaster;

        let probe = make_wkb("POINT (0 0)");
        if let Err(err) =
            crate::crs_utils::transform_wkb_to_crs(&probe, Some("EPSG:4326"), Some("EPSG:3857"))
        {
            let message = err.to_string();
            if message.contains("proj-sys") {
                return;
            }
            panic!("Unexpected CRS transform error: {message}");
        }

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let width = raster.metadata().width() as i64;
        let height = raster.metadata().height() as i64;
        let col = width / 2;
        let row = height / 2;
        let (lon, lat) = to_world_coordinate(&raster, col, row);

        let point_wkt = format!("POINT ({} {})", lon, lat);
        let point_wkb = make_wkb(&point_wkt);
        let (x_merc, y_merc) = web_mercator_from_lonlat(lon, lat);
        let point_merc_wkt = format!("POINT ({} {})", x_merc, y_merc);
        let point_merc_wkb = make_wkb(&point_merc_wkt);

        let raster_scalar = ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_array)));

        let geom_type_4326 = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:4326").unwrap());
        let geom_type_3857 = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap());

        let kernel = RsValuePoint { with_band: false };

        let result_4326 = kernel
            .invoke_batch(
                &[RASTER, geom_type_4326],
                &[
                    raster_scalar.clone(),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(point_wkb))),
                ],
            )
            .unwrap();

        let value_4326 = match result_4326 {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => value,
            _ => panic!("Expected Float64 scalar result"),
        };

        let result_3857 = kernel
            .invoke_batch(
                &[RASTER, geom_type_3857],
                &[
                    raster_scalar,
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(point_merc_wkb))),
                ],
            )
            .unwrap();

        let value_3857 = match result_3857 {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => value,
            _ => panic!("Expected Float64 scalar result"),
        };

        assert_eq!(value_4326, value_3857);
    }
}
