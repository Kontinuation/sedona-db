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

//! RS_MapAlgebra UDF - Apply a map algebra expression on raster(s)
//!
//! This function evaluates a mathematical expression for each pixel in the input raster(s)
//! and produces an output raster. The expression can reference input raster bands using
//! `rast[band_index]` syntax (or `rast0[band_index]` and `rast1[band_index]` for two-raster
//! operations).
//!
//! # Expression Syntax
//!
//! The expression evaluator supports standard mathematical operations:
//! - Arithmetic: `+`, `-`, `*`, `/`, `%` (modulo), `^` (power)
//! - Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
//! - Logic: `&&`, `||`, `!`
//! - Functions: `min`, `max`, `abs`, `sqrt`, `sin`, `cos`, `tan`, `ln`, `log`, `exp`, `floor`, `ceil`, `round`
//! - Conditionals: `if(condition, true_value, false_value)`
//!
//! # Variables
//!
//! For single-raster operations:
//! - `rast` or `rast0`, `rast1`, ..., `rastN`: Band values (where N is band index, 0-based)
//!
//! For two-raster operations:
//! - `rast0_0`, `rast0_1`, ...: First raster's band values
//! - `rast1_0`, `rast1_1`, ...: Second raster's band values
//!
//! Additional variables:
//! - `x`: Current pixel column (0-based)
//! - `y`: Current pixel row (0-based)
//! - `width`: Raster width
//! - `height`: Raster height

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray};
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use evalexpr::{build_operator_tree, ContextWithMutableVariables, HashMapContext, Value};

use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::{RasterRefImpl, RasterStructArray};
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::nodata_f64_to_bytes;
use crate::raster_band_reader::RasterBandReader;

/// RS_MapAlgebra() scalar UDF implementation
///
/// Apply a map algebra expression on raster(s)
pub fn rs_map_algebra_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_mapalgebra",
        vec![
            // Single raster variants
            Arc::new(RsMapAlgebra {
                two_raster: false,
                with_nodata: false,
                with_num_bands: false,
            }),
            Arc::new(RsMapAlgebra {
                two_raster: false,
                with_nodata: true,
                with_num_bands: false,
            }),
            Arc::new(RsMapAlgebra {
                two_raster: false,
                with_nodata: true,
                with_num_bands: true,
            }),
            // Two raster variants
            Arc::new(RsMapAlgebra {
                two_raster: true,
                with_nodata: false,
                with_num_bands: false,
            }),
            Arc::new(RsMapAlgebra {
                two_raster: true,
                with_nodata: true,
                with_num_bands: false,
            }),
            Arc::new(RsMapAlgebra {
                two_raster: true,
                with_nodata: true,
                with_num_bands: true,
            }),
        ],
        Volatility::Immutable,
        Some(rs_map_algebra_doc()),
    )
}

fn rs_map_algebra_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Applies a map algebra expression on raster(s). The expression is evaluated for each pixel.".to_string(),
        "RS_MapAlgebra(raster, pixelType, script) or RS_MapAlgebra(raster, pixelType, script, noDataValue, numBands)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster (use rast0, rast1, etc. for band values)")
    .with_argument("raster2", "Raster: Second input raster for two-raster operations (optional)")
    .with_argument("pixelType", "String: Output pixel type ('B'=UInt8, 'S'=Int16, 'I'=Int32, 'F'=Float32, 'D'=Float64)")
    .with_argument("script", "String: Map algebra expression (e.g., 'rast0 * 2 + rast1')")
    .with_argument("noDataValue", "Double: NoData value for output (optional)")
    .with_argument("numBands", "Integer: Number of output bands (optional, default: 1)")
    .with_sql_example("SELECT RS_MapAlgebra(rast, 'D', '(rast3 - rast0) / (rast3 + rast0)') AS ndvi FROM raster_table".to_string())
    .build()
}

/// Kernel implementation for RS_MapAlgebra
#[derive(Debug)]
struct RsMapAlgebra {
    two_raster: bool,
    with_nodata: bool,
    with_num_bands: bool,
}

impl SedonaScalarKernel for RsMapAlgebra {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = if self.two_raster {
            if self.with_num_bands {
                vec![
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_string(),
                    ArgMatcher::is_string(),
                    ArgMatcher::is_numeric(),
                    ArgMatcher::is_integer(),
                ]
            } else if self.with_nodata {
                vec![
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_string(),
                    ArgMatcher::is_string(),
                    ArgMatcher::is_numeric(),
                ]
            } else {
                vec![
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_string(),
                    ArgMatcher::is_string(),
                ]
            }
        } else if self.with_num_bands {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
                ArgMatcher::is_string(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_integer(),
            ]
        } else if self.with_nodata {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
                ArgMatcher::is_string(),
                ArgMatcher::is_numeric(),
            ]
        } else {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
                ArgMatcher::is_string(),
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
        let (pixel_type_idx, script_idx, nodata_idx, num_bands_idx) = if self.two_raster {
            if self.with_num_bands {
                (2, 3, Some(4), Some(5))
            } else if self.with_nodata {
                (2, 3, Some(4), None)
            } else {
                (2, 3, None, None)
            }
        } else if self.with_num_bands {
            (1, 2, Some(3), Some(4))
        } else if self.with_nodata {
            (1, 2, Some(3), None)
        } else {
            (1, 2, None, None)
        };

        // Get pixel type
        let pixel_type_str = extract_string_scalar(&args[pixel_type_idx])?
            .ok_or_else(|| DataFusionError::Execution("Pixel type is required".to_string()))?;
        let output_type = parse_pixel_type(&pixel_type_str)?;

        // Get script
        let script = extract_string_scalar(&args[script_idx])?
            .ok_or_else(|| DataFusionError::Execution("Script is required".to_string()))?;

        // Get nodata value
        let nodata = nodata_idx.and_then(|idx| extract_f64_scalar(&args[idx]).ok().flatten());

        // Get number of output bands
        let num_bands = num_bands_idx
            .and_then(|idx| extract_i32_scalar(&args[idx]).ok().flatten())
            .map(|n| n as usize)
            .unwrap_or(1);

        // Get raster arrays
        let raster_array0 = get_raster_array(&args[0])?;
        let raster_array1 = if self.two_raster {
            Some(get_raster_array(&args[1])?)
        } else {
            None
        };

        // Precompile the expression
        let compiled_expr = build_operator_tree(&script).map_err(|e| {
            DataFusionError::Execution(format!("Failed to parse expression '{}': {}", script, e))
        })?;

        // Build output rasters
        let mut builder = RasterBuilder::new(num_iterations);

        for i in 0..num_iterations {
            let raster_idx0 = if raster_array0.len() == 1 { 0 } else { i };

            if raster_array0.is_null(raster_idx0) {
                builder.append_null()?;
                continue;
            }

            let raster0 = raster_array0.get(raster_idx0)?;

            // Get second raster if two-raster operation
            let raster1 = if let Some(ref arr1) = raster_array1 {
                let raster_idx1 = if arr1.len() == 1 { 0 } else { i };
                if arr1.is_null(raster_idx1) {
                    builder.append_null()?;
                    continue;
                }
                Some(arr1.get(raster_idx1)?)
            } else {
                None
            };

            match apply_map_algebra(
                &raster0,
                raster1.as_ref(),
                &compiled_expr,
                &output_type,
                nodata,
                num_bands,
            ) {
                Ok(result_data) => {
                    build_result_raster(&mut builder, &raster0, &result_data)?;
                }
                Err(e) => {
                    eprintln!("RS_MapAlgebra error: {}", e);
                    builder.append_null()?;
                }
            }
        }

        let result = Arc::new(builder.finish()?) as ArrayRef;
        finish_result(args, result)
    }
}

/// Output data from map algebra operation
struct MapAlgebraResult {
    band_data: Vec<Vec<u8>>,
    band_metadata: Vec<BandMetadata>,
}

/// Parse pixel type string to BandDataType
fn parse_pixel_type(pixel_type: &str) -> Result<BandDataType> {
    match pixel_type.to_uppercase().as_str() {
        "B" | "BYTE" | "UINT8" => Ok(BandDataType::UInt8),
        "S" | "SHORT" | "INT16" => Ok(BandDataType::Int16),
        "US" | "USHORT" | "UINT16" => Ok(BandDataType::UInt16),
        "I" | "INT" | "INT32" => Ok(BandDataType::Int32),
        "UI" | "UINT" | "UINT32" => Ok(BandDataType::UInt32),
        "F" | "FLOAT" | "FLOAT32" => Ok(BandDataType::Float32),
        "D" | "DOUBLE" | "FLOAT64" => Ok(BandDataType::Float64),
        _ => Err(DataFusionError::Execution(format!(
            "Unknown pixel type '{}'. Use: B(yte), S(hort), I(nt), F(loat), D(ouble)",
            pixel_type
        ))),
    }
}

/// Apply map algebra expression to raster(s)
fn apply_map_algebra(
    raster0: &RasterRefImpl<'_>,
    raster1: Option<&RasterRefImpl<'_>>,
    expr: &evalexpr::Node,
    output_type: &BandDataType,
    nodata: Option<f64>,
    num_bands: usize,
) -> Result<MapAlgebraResult> {
    let metadata = raster0.metadata();
    let width = metadata.width() as usize;
    let height = metadata.height() as usize;
    let pixel_count = width * height;

    // Read all band data from first raster
    let bands0 = raster0.bands();
    let mut reader0 = RasterBandReader::new(raster0);
    let band_data0: Vec<Vec<f64>> = (1..=bands0.len())
        .map(|i| reader0.read_band_f64(i))
        .collect::<Result<Vec<_>>>()?;

    // Read all band data from second raster (if present)
    let band_data1: Option<Vec<Vec<f64>>> = if let Some(r1) = raster1 {
        // Validate dimensions match
        let m1 = r1.metadata();
        if m1.width() != metadata.width() || m1.height() != metadata.height() {
            return Err(DataFusionError::Execution(
                "Raster dimensions must match for two-raster map algebra".to_string(),
            ));
        }
        let bands1 = r1.bands();
        let mut reader1 = RasterBandReader::new(r1);
        Some(
            (1..=bands1.len())
                .map(|i| reader1.read_band_f64(i))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };

    // Allocate output band data
    let byte_size = data_type_byte_size(output_type);
    let mut output_bands: Vec<Vec<u8>> = (0..num_bands)
        .map(|_| vec![0u8; pixel_count * byte_size])
        .collect();

    // Determine nodata value
    let nodata_val = nodata.unwrap_or(0.0);

    // Create evaluation context
    let mut context = HashMapContext::new();

    // Set constant variables
    context
        .set_value("width".to_string(), Value::Float(width as f64))
        .map_err(|e| DataFusionError::Execution(format!("Failed to set width: {}", e)))?;
    context
        .set_value("height".to_string(), Value::Float(height as f64))
        .map_err(|e| DataFusionError::Execution(format!("Failed to set height: {}", e)))?;

    // Evaluate expression for each pixel
    for pixel_idx in 0..pixel_count {
        let x = pixel_idx % width;
        let y = pixel_idx / width;

        // Set position variables
        context
            .set_value("x".to_string(), Value::Float(x as f64))
            .map_err(|e| DataFusionError::Execution(format!("Failed to set x: {}", e)))?;
        context
            .set_value("y".to_string(), Value::Float(y as f64))
            .map_err(|e| DataFusionError::Execution(format!("Failed to set y: {}", e)))?;

        // Set band values for first raster
        // Support both rast0, rast1, ... and rast0_0, rast0_1, ... syntax
        for (band_idx, band_values) in band_data0.iter().enumerate() {
            let value = band_values[pixel_idx];
            // rast0, rast1, rast2, ... (single raster syntax)
            context
                .set_value(format!("rast{}", band_idx), Value::Float(value))
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to set rast{}: {}", band_idx, e))
                })?;
            // rast0_0, rast0_1, ... (two-raster syntax, first raster)
            context
                .set_value(format!("rast0_{}", band_idx), Value::Float(value))
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to set rast0_{}: {}", band_idx, e))
                })?;
        }

        // Set band values for second raster (if present)
        if let Some(ref bands1) = band_data1 {
            for (band_idx, band_values) in bands1.iter().enumerate() {
                let value = band_values[pixel_idx];
                context
                    .set_value(format!("rast1_{}", band_idx), Value::Float(value))
                    .map_err(|e| {
                        DataFusionError::Execution(format!(
                            "Failed to set rast1_{}: {}",
                            band_idx, e
                        ))
                    })?;
            }
        }

        // Evaluate expression
        let result = expr.eval_with_context(&context).map_err(|e| {
            DataFusionError::Execution(format!(
                "Expression evaluation failed at pixel ({}, {}): {}",
                x, y, e
            ))
        })?;

        // Handle the result based on number of output bands
        if num_bands == 1 {
            // Single output band - use the result directly
            let value = value_to_f64(&result)?;
            write_pixel_value(&mut output_bands[0], pixel_idx, output_type, value);
        } else {
            // Multiple output bands - expect a tuple result or set all bands to same value
            match result {
                Value::Tuple(values) => {
                    for (band_idx, val) in values.iter().enumerate().take(num_bands) {
                        let value = value_to_f64(val)?;
                        write_pixel_value(
                            &mut output_bands[band_idx],
                            pixel_idx,
                            output_type,
                            value,
                        );
                    }
                    // If tuple has fewer values than bands, fill remaining with nodata
                    for band in output_bands.iter_mut().take(num_bands).skip(values.len()) {
                        write_pixel_value(band, pixel_idx, output_type, nodata_val);
                    }
                }
                _ => {
                    // Single value - apply to first band, nodata for rest
                    let value = value_to_f64(&result)?;
                    write_pixel_value(&mut output_bands[0], pixel_idx, output_type, value);
                    for band in output_bands.iter_mut().take(num_bands).skip(1) {
                        write_pixel_value(band, pixel_idx, output_type, nodata_val);
                    }
                }
            }
        }
    }

    // Build band metadata
    let band_metadata: Vec<BandMetadata> = (0..num_bands)
        .map(|_| BandMetadata {
            nodata_value: nodata.map(|v| nodata_f64_to_bytes(v, output_type)),
            storage_type: StorageType::InDb,
            datatype: output_type.clone(),
            outdb_url: None,
            outdb_band_id: None,
        })
        .collect();

    Ok(MapAlgebraResult {
        band_data: output_bands,
        band_metadata,
    })
}

/// Convert evalexpr Value to f64
fn value_to_f64(value: &Value) -> Result<f64> {
    match value {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        Value::Boolean(b) => Ok(if *b { 1.0 } else { 0.0 }),
        _ => Err(DataFusionError::Execution(format!(
            "Cannot convert {:?} to numeric value",
            value
        ))),
    }
}

/// Write a pixel value to band data
fn write_pixel_value(data: &mut [u8], offset: usize, data_type: &BandDataType, value: f64) {
    let byte_size = data_type_byte_size(data_type);
    let byte_offset = offset * byte_size;

    match data_type {
        BandDataType::UInt8 => {
            data[byte_offset] = value.clamp(0.0, 255.0) as u8;
        }
        BandDataType::UInt16 => {
            let v = value.clamp(0.0, u16::MAX as f64) as u16;
            data[byte_offset..byte_offset + 2].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::Int16 => {
            let v = value.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
            data[byte_offset..byte_offset + 2].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::UInt32 => {
            let v = value.clamp(0.0, u32::MAX as f64) as u32;
            data[byte_offset..byte_offset + 4].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::Int32 => {
            let v = value.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            data[byte_offset..byte_offset + 4].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::Float32 => {
            let v = value as f32;
            data[byte_offset..byte_offset + 4].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::Float64 => {
            data[byte_offset..byte_offset + 8].copy_from_slice(&value.to_le_bytes());
        }
    }
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

/// Build result raster using RasterBuilder
fn build_result_raster(
    builder: &mut RasterBuilder,
    original_raster: &RasterRefImpl<'_>,
    result: &MapAlgebraResult,
) -> Result<()> {
    let original_metadata = original_raster.metadata();

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

    for (band_data, band_metadata) in result.band_data.iter().zip(result.band_metadata.iter()) {
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
        ColumnarValue::Scalar(ScalarValue::Null) => Ok(None),
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
    fn test_parse_pixel_type() {
        assert_eq!(parse_pixel_type("B").unwrap(), BandDataType::UInt8);
        assert_eq!(parse_pixel_type("byte").unwrap(), BandDataType::UInt8);
        assert_eq!(parse_pixel_type("S").unwrap(), BandDataType::Int16);
        assert_eq!(parse_pixel_type("I").unwrap(), BandDataType::Int32);
        assert_eq!(parse_pixel_type("F").unwrap(), BandDataType::Float32);
        assert_eq!(parse_pixel_type("D").unwrap(), BandDataType::Float64);
        assert_eq!(parse_pixel_type("FLOAT64").unwrap(), BandDataType::Float64);
        assert!(parse_pixel_type("X").is_err());
    }

    #[test]
    fn test_value_to_f64() {
        let pi = std::f64::consts::PI;
        assert!((value_to_f64(&Value::Float(pi)).unwrap() - pi).abs() < f64::EPSILON);
        assert_eq!(value_to_f64(&Value::Int(42)).unwrap(), 42.0);
        assert_eq!(value_to_f64(&Value::Boolean(true)).unwrap(), 1.0);
        assert_eq!(value_to_f64(&Value::Boolean(false)).unwrap(), 0.0);
    }

    #[test]
    fn test_write_pixel_value() {
        let mut data = vec![0u8; 8];

        // Test UInt8
        write_pixel_value(&mut data, 0, &BandDataType::UInt8, 128.0);
        assert_eq!(data[0], 128);

        // Test Float64
        let mut data64 = vec![0u8; 8];
        let pi = std::f64::consts::PI;
        write_pixel_value(&mut data64, 0, &BandDataType::Float64, pi);
        let read_back = f64::from_le_bytes([
            data64[0], data64[1], data64[2], data64[3], data64[4], data64[5], data64[6], data64[7],
        ]);
        assert!((read_back - pi).abs() < 1e-10);
    }

    #[test]
    fn test_expression_evaluation() {
        let expr = build_operator_tree("rast0 * 2 + 1").unwrap();
        let mut context = HashMapContext::new();
        context
            .set_value("rast0".to_string(), Value::Float(10.0))
            .unwrap();
        let result = expr.eval_with_context(&context).unwrap();
        assert_eq!(value_to_f64(&result).unwrap(), 21.0);
    }

    #[test]
    fn test_ndvi_expression() {
        // NDVI = (NIR - Red) / (NIR + Red)
        let expr = build_operator_tree("(rast3 - rast0) / (rast3 + rast0)").unwrap();
        let mut context = HashMapContext::new();
        // Simulate: Red=100, NIR=200
        context
            .set_value("rast0".to_string(), Value::Float(100.0))
            .unwrap();
        context
            .set_value("rast3".to_string(), Value::Float(200.0))
            .unwrap();
        let result = expr.eval_with_context(&context).unwrap();
        let ndvi = value_to_f64(&result).unwrap();
        // NDVI = (200-100)/(200+100) = 100/300 = 0.333...
        assert!((ndvi - 0.333333).abs() < 0.001);
    }

    #[test]
    fn test_map_algebra_with_test_raster() {
        use crate::rs_from_gdal_raster::RsFromGDALRaster;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();

        // Simple expression: multiply band 0 by 2
        let expr = build_operator_tree("rast0 * 2").unwrap();

        let result = apply_map_algebra(&raster, None, &expr, &BandDataType::Float64, None, 1);
        assert!(
            result.is_ok(),
            "Map algebra should succeed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert_eq!(output.band_data.len(), 1);

        // Verify output size matches input
        let metadata = raster.metadata();
        let expected_size =
            (metadata.width() * metadata.height()) as usize * std::mem::size_of::<f64>();
        assert_eq!(output.band_data[0].len(), expected_size);
    }

    #[test]
    fn test_map_algebra_multi_band_output() {
        use crate::rs_from_gdal_raster::RsFromGDALRaster;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();

        // Expression that produces single value (will be applied to first band only)
        let expr = build_operator_tree("rast0 + rast0").unwrap();

        let result = apply_map_algebra(&raster, None, &expr, &BandDataType::Float32, Some(0.0), 2);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.band_data.len(), 2);
        assert_eq!(output.band_metadata.len(), 2);
    }
}
