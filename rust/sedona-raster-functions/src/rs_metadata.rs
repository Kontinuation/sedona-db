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
use std::{sync::Arc, vec};

use crate::executor::RasterExecutor;
use arrow_array::builder::{Float64Builder, Int32Builder, UInt64Builder};
use arrow_array::StructArray;
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::crs::deserialize_crs;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_MetaData() scalar UDF implementation
///
/// Returns the metadata of the raster as a struct
pub fn rs_metadata_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_metadata",
        vec![Arc::new(RsMetaData {})],
        Volatility::Immutable,
        Some(rs_metadata_doc()),
    )
}

fn rs_metadata_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the metadata of the raster as a struct containing: upperLeftX, upperLeftY, gridWidth, gridHeight, scaleX, scaleY, skewX, skewY, srid, numSampleDimensions, tileWidth, tileHeight.".to_string(),
        "RS_MetaData(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_MetaData(RS_Example())".to_string())
    .build()
}

/// Returns the schema for the metadata struct
fn metadata_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("upperLeftX", DataType::Float64, true),
        Field::new("upperLeftY", DataType::Float64, true),
        Field::new("gridWidth", DataType::UInt64, true),
        Field::new("gridHeight", DataType::UInt64, true),
        Field::new("scaleX", DataType::Float64, true),
        Field::new("scaleY", DataType::Float64, true),
        Field::new("skewX", DataType::Float64, true),
        Field::new("skewY", DataType::Float64, true),
        Field::new("srid", DataType::Int32, true),
        Field::new("numSampleDimensions", DataType::UInt64, true),
        Field::new("tileWidth", DataType::UInt64, true),
        Field::new("tileHeight", DataType::UInt64, true),
    ])
}

#[derive(Debug)]
struct RsMetaData {}

impl SedonaScalarKernel for RsMetaData {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Struct(metadata_struct_fields())),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let capacity = executor.num_iterations();

        let mut upper_left_x_builder = Float64Builder::with_capacity(capacity);
        let mut upper_left_y_builder = Float64Builder::with_capacity(capacity);
        let mut grid_width_builder = UInt64Builder::with_capacity(capacity);
        let mut grid_height_builder = UInt64Builder::with_capacity(capacity);
        let mut scale_x_builder = Float64Builder::with_capacity(capacity);
        let mut scale_y_builder = Float64Builder::with_capacity(capacity);
        let mut skew_x_builder = Float64Builder::with_capacity(capacity);
        let mut skew_y_builder = Float64Builder::with_capacity(capacity);
        let mut srid_builder = Int32Builder::with_capacity(capacity);
        let mut num_bands_builder = UInt64Builder::with_capacity(capacity);
        let mut tile_width_builder = UInt64Builder::with_capacity(capacity);
        let mut tile_height_builder = UInt64Builder::with_capacity(capacity);

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                None => {
                    upper_left_x_builder.append_null();
                    upper_left_y_builder.append_null();
                    grid_width_builder.append_null();
                    grid_height_builder.append_null();
                    scale_x_builder.append_null();
                    scale_y_builder.append_null();
                    skew_x_builder.append_null();
                    skew_y_builder.append_null();
                    srid_builder.append_null();
                    num_bands_builder.append_null();
                    tile_width_builder.append_null();
                    tile_height_builder.append_null();
                }
                Some(raster) => {
                    let metadata = raster.metadata();

                    upper_left_x_builder.append_value(metadata.upper_left_x());
                    upper_left_y_builder.append_value(metadata.upper_left_y());
                    grid_width_builder.append_value(metadata.width());
                    grid_height_builder.append_value(metadata.height());
                    scale_x_builder.append_value(metadata.scale_x());
                    scale_y_builder.append_value(metadata.scale_y());
                    skew_x_builder.append_value(metadata.skew_x());
                    skew_y_builder.append_value(metadata.skew_y());

                    // Extract SRID from CRS
                    let srid = match raster.crs() {
                        None => 0i32,
                        Some(crs_str) => {
                            let crs = deserialize_crs(crs_str).map_err(|e| {
                                DataFusionError::Execution(format!(
                                    "Failed to deserialize CRS: {e}"
                                ))
                            })?;

                            match crs {
                                Some(crs_ref) => {
                                    let srid_opt = crs_ref.srid().map_err(|e| {
                                        DataFusionError::Execution(format!(
                                            "Failed to get SRID from CRS: {e}"
                                        ))
                                    })?;
                                    srid_opt.map(|s| s as i32).unwrap_or(0)
                                }
                                None => 0i32,
                            }
                        }
                    };
                    srid_builder.append_value(srid);

                    num_bands_builder.append_value(raster.bands().len() as u64);

                    // For tile dimensions, we use the full raster dimensions
                    // as our rasters are not internally tiled
                    tile_width_builder.append_value(metadata.width());
                    tile_height_builder.append_value(metadata.height());
                }
            }
            Ok(())
        })?;

        let struct_array = StructArray::new(
            metadata_struct_fields(),
            vec![
                Arc::new(upper_left_x_builder.finish()),
                Arc::new(upper_left_y_builder.finish()),
                Arc::new(grid_width_builder.finish()),
                Arc::new(grid_height_builder.finish()),
                Arc::new(scale_x_builder.finish()),
                Arc::new(scale_y_builder.finish()),
                Arc::new(skew_x_builder.finish()),
                Arc::new(skew_y_builder.finish()),
                Arc::new(srid_builder.finish()),
                Arc::new(num_bands_builder.finish()),
                Arc::new(tile_width_builder.finish()),
                Arc::new(tile_height_builder.finish()),
            ],
            None,
        );

        executor.finish(Arc::new(struct_array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{cast::AsArray, Array};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_metadata_udf().into();
        assert_eq!(udf.name(), "rs_metadata");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn udf_metadata_invoke() {
        let udf: ScalarUDF = rs_metadata_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        // Test with rasters
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();

        let struct_array = result.as_struct();
        assert_eq!(struct_array.len(), 3);

        // Check first raster (index 0) metadata
        let upper_left_x = struct_array
            .column(0)
            .as_primitive::<arrow_array::types::Float64Type>();
        assert_eq!(upper_left_x.value(0), 1.0);
        assert!(upper_left_x.is_null(1)); // null raster
        assert_eq!(upper_left_x.value(2), 3.0);

        let grid_width = struct_array
            .column(2)
            .as_primitive::<arrow_array::types::UInt64Type>();
        assert_eq!(grid_width.value(0), 1);
        assert!(grid_width.is_null(1)); // null raster
        assert_eq!(grid_width.value(2), 3);

        let num_bands = struct_array
            .column(9)
            .as_primitive::<arrow_array::types::UInt64Type>();
        assert_eq!(num_bands.value(0), 1);
        assert!(num_bands.is_null(1)); // null raster
        assert_eq!(num_bands.value(2), 1);

        // Check SRID - generate_test_rasters uses OGC:CRS84 which maps to 4326
        let srid = struct_array
            .column(8)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(srid.value(0), 4326);
        assert!(srid.is_null(1)); // null raster
        assert_eq!(srid.value(2), 4326);
    }

    #[test]
    fn udf_metadata_null_scalar() {
        let udf: ScalarUDF = rs_metadata_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        // Test with null scalar
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        match result {
            ScalarValue::Struct(struct_arr) => {
                // All fields should be null
                for col in struct_arr.columns() {
                    assert!(col.is_null(0));
                }
            }
            _ => panic!("Expected scalar struct result"),
        }
    }
}
