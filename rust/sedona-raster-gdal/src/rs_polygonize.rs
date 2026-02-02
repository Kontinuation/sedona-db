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

//! RS_Polygonize UDF - Convert raster band to vector polygons
//!
//! Returns a list of polygons for all connected regions of pixels with the same
//! value in the specified band.
use std::convert::TryInto;
use std::sync::Arc;

use arrow_array::builder::{BinaryBuilder, Float64Builder, ListBuilder, StructBuilder};
use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use gdal::raster::PolygonizeOptions;
use gdal::vector::LayerAccess;
use gdal::vector::{OGRFieldType, OGRwkbGeometryType};
use gdal::DriverManager;

use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::{RasterRefImpl, RasterStructArray};
use sedona_raster::traits::RasterRef;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

// `dataset` removed; the provider is used instead when creating GDAL datasets.
use crate::gdal_dataset_provider::configure_thread_local_cache_size;

/// RS_Polygonize() scalar UDF implementation
///
/// Returns a list of polygons for connected regions of pixels with the same value
pub fn rs_polygonize_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_polygonize",
        vec![Arc::new(RsPolygonize)],
        Volatility::Immutable,
        Some(rs_polygonize_doc()),
    )
}

fn rs_polygonize_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns a list of polygons for all connected regions of pixels with the same value in the specified band.".to_string(),
        "RS_Polygonize(raster: Raster, band: Integer)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("band", "Integer: Band number (1-based)")
    .with_sql_example("SELECT explode(RS_Polygonize(raster, 1)) FROM raster_table".to_string())
    .build()
}

/// Kernel implementation for RS_Polygonize
#[derive(Debug)]
struct RsPolygonize;

impl SedonaScalarKernel for RsPolygonize {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            // Return type is List<Struct<geom: Binary, value: Float64>>
            SedonaType::Arrow(polygon_value_list_type()),
        );
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

        // Get the band number
        let band_num = extract_i32_scalar(&args[1])?
            .unwrap_or(1)
            .max(1)
            .try_into()
            .unwrap_or(1);

        // Get raster array
        let raster_array = get_raster_array(&args[0])?;

        // Build result as List<Struct<geom, value>>
        let struct_fields = polygon_value_struct_fields();
        let mut list_builder = ListBuilder::new(StructBuilder::from_fields(struct_fields, 16));

        for i in 0..num_iterations {
            let raster_idx = if raster_array.len() == 1 { 0 } else { i };

            if raster_array.is_null(raster_idx) {
                list_builder.append_null();
                continue;
            }

            let raster = raster_array.get(raster_idx)?;

            match polygonize_raster(&raster, band_num) {
                Ok(polygon_values) => {
                    let struct_builder = list_builder.values();

                    for (wkb, value) in polygon_values {
                        // Get field builders
                        let geom_builder = struct_builder
                            .field_builder::<BinaryBuilder>(0)
                            .expect("Expected BinaryBuilder for geom field");
                        geom_builder.append_value(&wkb);

                        let value_builder = struct_builder
                            .field_builder::<Float64Builder>(1)
                            .expect("Expected Float64Builder for value field");
                        value_builder.append_value(value);

                        struct_builder.append(true);
                    }
                    list_builder.append(true);
                }
                Err(e) => {
                    // Log error but append null
                    eprintln!("Polygonize error: {}", e);
                    list_builder.append_null();
                }
            }
        }

        let result = Arc::new(list_builder.finish()) as ArrayRef;
        finish_result(args, result)
    }
}

/// Return type for the list of polygon/value structs
fn polygon_value_list_type() -> DataType {
    DataType::List(Arc::new(Field::new(
        "item",
        DataType::Struct(polygon_value_struct_fields()),
        true,
    )))
}

/// Struct fields for polygon/value pairs
fn polygon_value_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("geom", DataType::Binary, false),
        Field::new("value", DataType::Float64, false),
    ])
}

/// Polygonize a raster band using GDAL
fn polygonize_raster(raster: &RasterRefImpl<'_>, band_num: usize) -> Result<Vec<(Vec<u8>, f64)>> {
    let bands = raster.bands();
    if band_num == 0 || band_num > bands.len() {
        return Err(DataFusionError::Execution(format!(
            "Band {} is out of range (1-{})",
            band_num,
            bands.len()
        )));
    }

    // Create GDAL dataset from raster (thread-local provider)
    let provider = crate::gdal_dataset_provider::thread_local_provider()
        .map_err(|e| DataFusionError::Execution(format!("Failed to init GDAL provider: {}", e)))?;
    let raster_ds = provider
        .raster_ref_to_gdal(raster)
        .map_err(|e| DataFusionError::Execution(format!("Failed to create GDAL dataset: {}", e)))?;
    let gdal_dataset = raster_ds.as_dataset();

    // Get the raster band
    let raster_band = gdal_dataset.rasterband(band_num).map_err(|e| {
        DataFusionError::Execution(format!("Failed to get band {}: {}", band_num, e))
    })?;

    // Create memory datasource for output polygons
    let mem_driver = DriverManager::get_driver_by_name("MEM")
        .map_err(|e| DataFusionError::Execution(format!("Failed to get MEM driver: {}", e)))?;

    let mut mem_ds = mem_driver.create_vector_only("").map_err(|e| {
        DataFusionError::Execution(format!("Failed to create memory dataset: {}", e))
    })?;

    // Create layer with geometry field and value field
    let spatial_ref = gdal_dataset.spatial_ref().ok();
    let layer = mem_ds
        .create_layer(gdal::vector::LayerOptions {
            name: "polygons",
            srs: spatial_ref.as_ref(),
            ty: OGRwkbGeometryType::wkbPolygon,
            options: None,
        })
        .map_err(|e| DataFusionError::Execution(format!("Failed to create layer: {}", e)))?;

    // Add pixel value field
    let field_defn = gdal::vector::FieldDefn::new("value", OGRFieldType::OFTReal).map_err(|e| {
        DataFusionError::Execution(format!("Failed to create field definition: {}", e))
    })?;
    field_defn
        .add_to_layer(&layer)
        .map_err(|e| DataFusionError::Execution(format!("Failed to add field to layer: {}", e)))?;

    // Call GDAL Polygonize via georust/gdal safe wrapper.
    let polygonize_options = PolygonizeOptions::new();
    gdal::raster::polygonize(&raster_band, None, &layer, 0, &polygonize_options)
        .map_err(|e| DataFusionError::Execution(format!("GDAL polygonize failed: {e}")))?;

    // Extract polygons from layer
    let mut polygon_values = Vec::new();

    let mut value_field_idx: Option<usize> = None;
    let mut layer_for_read = layer;
    for feature in layer_for_read.features() {
        let geom = feature.geometry().ok_or_else(|| {
            DataFusionError::Execution("Polygonize output feature missing geometry".to_string())
        })?;
        let wkb = geom.iso_wkb().map_err(|e| {
            DataFusionError::Execution(format!("Failed to export geometry to WKB: {e}"))
        })?;

        let idx = match value_field_idx {
            Some(idx) => idx,
            None => {
                let idx = feature.field_index("value").map_err(|e| {
                    DataFusionError::Execution(format!("Missing 'value' field: {e}"))
                })?;
                value_field_idx = Some(idx);
                idx
            }
        };

        let value = feature
            .field_as_double(idx)
            .map_err(|e| DataFusionError::Execution(format!("Failed to read 'value' field: {e}")))?
            .unwrap_or(0.0);

        polygon_values.push((wkb, value));
    }

    Ok(polygon_values)
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
    Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(&out, 0)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_value_list_type() {
        let dt = polygon_value_list_type();
        match dt {
            DataType::List(field) => {
                assert_eq!(field.name(), "item");
                match field.data_type() {
                    DataType::Struct(fields) => {
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].name(), "geom");
                        assert_eq!(fields[1].name(), "value");
                    }
                    _ => panic!("Expected Struct data type"),
                }
            }
            _ => panic!("Expected List data type"),
        }
    }

    #[test]
    fn test_polygonize_raster() {
        // Load test raster and polygonize it
        use crate::rs_from_gdal_raster::RsFromGDALRaster;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        // Create a RasterStructArray to access the raster
        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();

        // Polygonize band 1
        let result = polygonize_raster(&raster, 1).unwrap();

        // Should return at least one polygon
        assert!(
            !result.is_empty(),
            "Polygonize should return at least one polygon"
        );

        // Each result should have a valid WKB and value
        for (wkb, value) in &result {
            // WKB should be at least 5 bytes (header)
            assert!(wkb.len() >= 5, "WKB should be at least 5 bytes");
            // Value should be finite
            assert!(value.is_finite(), "Value should be a finite number");
        }
    }

    #[test]
    fn test_polygonize_kernel_return_type() {
        use arrow_schema::DataType;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_schema::datatypes::RASTER;

        let kernel = RsPolygonize;

        let arg_types = vec![RASTER, SedonaType::Arrow(DataType::Int32)];
        let return_type = kernel.return_type(&arg_types).unwrap();
        assert!(return_type.is_some());

        // Return type should be List<Struct<geom, value>>
        match return_type.unwrap() {
            SedonaType::Arrow(DataType::List(_)) => (),
            _ => panic!("Expected List return type"),
        }
    }

    #[test]
    fn test_polygonize_invoke_batch() {
        use crate::rs_from_gdal_raster::RsFromGDALRaster;
        use arrow_schema::DataType;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_schema::datatypes::RASTER;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let content = std::fs::read(&test_file).unwrap();
        let raster_array = RsFromGDALRaster::parse_gdal_raster(&content).unwrap();

        let kernel = RsPolygonize;

        let arg_types = vec![RASTER, SedonaType::Arrow(DataType::Int32)];
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_array))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(1))), // band
        ];

        let result = kernel
            .invoke_batch_from_args(
                &arg_types,
                &args,
                &SedonaType::Arrow(DataType::Null),
                0,
                None,
            )
            .unwrap();

        // Result should be a scalar (since input was scalar)
        match result {
            ColumnarValue::Scalar(_) => (),
            _ => panic!("Expected Scalar result"),
        }
    }
}
