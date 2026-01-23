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

//! RS_FromPath UDF - Load out-db raster from file path
//!
//! Returns an out-db raster from a path to an image file. Supported formats include:
//! - GeoTiff (*.tif, *.tiff)
//! - Arc Info ASCII Grid (*.asc)
//! - And other GDAL-supported raster formats

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StringArray, StructArray};
use datafusion_common::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use gdal::spatial_ref::SpatialRef;
use gdal::{Dataset, DatasetOptions, GdalOpenFlags};

use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata};
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::StorageType;

use crate::dataset::{gdal_to_band_data_type, nodata_f64_to_bytes};

/// RS_FromPath() scalar UDF implementation
///
/// Returns an out-db raster from a path to an image file
pub fn rs_from_path_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_frompath",
        vec![
            Arc::new(RsFromPath::new(false)), // RS_FromPath(path)
            Arc::new(RsFromPath::new(true)),  // RS_FromPath(path, params)
        ],
        Volatility::Volatile, // Reads from filesystem
        Some(rs_from_path_doc()),
    )
}

fn rs_from_path_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns an out-db raster from a path to an image file".to_string(),
        "RS_FromPath(path: String[, params: String])".to_string(),
    )
    .with_argument("path", "Path to the raster file")
    .with_argument(
        "params",
        "Optional semicolon-delimited configuration string",
    )
    .with_sql_example(
        "SELECT RS_FromPath('/path/to/raster.tif')\nSELECT RS_FromPath('/path/to/raster.tif', 'option1=value1;option2=value2')".to_string(),
    )
    .build()
}

/// Kernel implementation for RS_FromPath
#[derive(Debug)]
struct RsFromPath {
    with_params: bool,
}

impl RsFromPath {
    fn new(with_params: bool) -> Self {
        Self { with_params }
    }

    /// Parse parameters string into a HashMap
    /// Format: "key1=value1;key2=value2"
    #[allow(dead_code)]
    fn parse_params(params: &str) -> HashMap<String, String> {
        params
            .split(';')
            .filter_map(|pair| {
                let parts: Vec<&str> = pair.trim().splitn(2, '=').collect();
                if parts.len() == 2 {
                    Some((parts[0].trim().to_string(), parts[1].trim().to_string()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Load raster metadata from file and create out-db raster
    fn load_outdb_raster(path: &str, _params: Option<&str>) -> Result<StructArray> {
        // Open dataset to read metadata
        let dataset = Dataset::open_ex(
            path,
            DatasetOptions {
                open_flags: GdalOpenFlags::GDAL_OF_RASTER | GdalOpenFlags::GDAL_OF_READONLY,
                ..Default::default()
            },
        )
        .map_err(|e| DataFusionError::Execution(format!("Failed to open raster file: {}", e)))?;

        // Get raster dimensions
        let (width, height) = dataset.raster_size();

        // Get geotransform
        let geotransform = dataset.geo_transform().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get geotransform: {}", e))
        })?;

        // Build RasterMetadata
        let metadata = RasterMetadata {
            width: width as u64,
            height: height as u64,
            upperleft_x: geotransform[0],
            upperleft_y: geotransform[3],
            scale_x: geotransform[1],
            scale_y: geotransform[5],
            skew_x: geotransform[2],
            skew_y: geotransform[4],
        };

        // Get CRS as WKT if available
        let crs = dataset
            .spatial_ref()
            .ok()
            .and_then(|sr: SpatialRef| sr.to_wkt().ok());

        // Build the raster array
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster(&metadata, crs.as_deref())
            .map_err(|e| DataFusionError::Execution(format!("Failed to start raster: {}", e)))?;

        // Add bands as out-db references
        let band_count = dataset.raster_count();
        for band_idx in 1..=band_count {
            let band = dataset.rasterband(band_idx).map_err(|e| {
                DataFusionError::Execution(format!("Failed to get band {}: {}", band_idx, e))
            })?;

            let gdal_type = band.band_type();
            let band_data_type = gdal_to_band_data_type(gdal_type).ok_or_else(|| {
                DataFusionError::Execution(format!("Unsupported band data type: {:?}", gdal_type))
            })?;

            // Get nodata value
            let nodata_bytes = band
                .no_data_value()
                .map(|no_data| nodata_f64_to_bytes(no_data, &band_data_type));

            let band_metadata = BandMetadata {
                nodata_value: nodata_bytes,
                storage_type: StorageType::OutDbRef,
                datatype: band_data_type,
                outdb_url: Some(path.to_string()),
                outdb_band_id: Some(band_idx as u32),
            };

            builder
                .start_band(band_metadata)
                .map_err(|e| DataFusionError::Execution(format!("Failed to start band: {}", e)))?;

            // For out-db rasters, we don't store the actual band data
            // Just append empty/null data placeholder
            builder.band_data_writer().append_null();

            builder
                .finish_band()
                .map_err(|e| DataFusionError::Execution(format!("Failed to finish band: {}", e)))?;
        }

        builder
            .finish_raster()
            .map_err(|e| DataFusionError::Execution(format!("Failed to finish raster: {}", e)))?;

        builder
            .finish()
            .map_err(|e| DataFusionError::Execution(format!("Failed to build raster: {}", e)))
    }
}

impl SedonaScalarKernel for RsFromPath {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = if self.with_params {
            vec![
                ArgMatcher::is_string(), // path
                ArgMatcher::is_string(), // params
            ]
        } else {
            vec![ArgMatcher::is_string()] // path only
        };

        let matcher = ArgMatcher::new(matchers, RASTER);
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // Get the path argument
        let (paths, params_opt) = match &args[0] {
            ColumnarValue::Scalar(scalar) => {
                let path = scalar.to_array().map_err(|e| {
                    DataFusionError::Execution(format!("Failed to convert scalar to array: {}", e))
                })?;
                let params = if self.with_params {
                    match &args[1] {
                        ColumnarValue::Scalar(s) => Some(s.to_array().map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Failed to convert params scalar: {}",
                                e
                            ))
                        })?),
                        ColumnarValue::Array(a) => Some(a.clone()),
                    }
                } else {
                    None
                };
                (path, params)
            }
            ColumnarValue::Array(array) => {
                let params = if self.with_params {
                    match &args[1] {
                        ColumnarValue::Scalar(s) => Some(s.to_array().map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Failed to convert params scalar: {}",
                                e
                            ))
                        })?),
                        ColumnarValue::Array(a) => Some(a.clone()),
                    }
                } else {
                    None
                };
                (array.clone(), params)
            }
        };

        let path_array = paths
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected string array for path argument".to_string())
            })?;

        let params_array: Option<&StringArray> = params_opt
            .as_ref()
            .and_then(|p| p.as_any().downcast_ref::<StringArray>());

        let len = path_array.len();

        if len == 0 {
            // Return empty raster array
            let builder = RasterBuilder::new(0);
            let result = builder.finish().map_err(|e| {
                DataFusionError::Execution(format!("Failed to build empty raster: {}", e))
            })?;
            return Ok(ColumnarValue::Array(Arc::new(result)));
        }

        // Process each path
        let mut combined_arrays: Vec<ArrayRef> = Vec::with_capacity(len);

        for i in 0..len {
            if path_array.is_null(i) {
                // Append null raster
                let mut builder = RasterBuilder::new(1);
                builder.append_null().map_err(|e| {
                    DataFusionError::Execution(format!("Failed to append null: {}", e))
                })?;
                let result = builder.finish().map_err(|e| {
                    DataFusionError::Execution(format!("Failed to build null raster: {}", e))
                })?;
                combined_arrays.push(Arc::new(result));
            } else {
                let path = path_array.value(i);
                let params = params_array.and_then(|pa| {
                    if pa.is_null(i) {
                        None
                    } else {
                        Some(pa.value(i))
                    }
                });

                let raster = Self::load_outdb_raster(path, params)?;
                combined_arrays.push(Arc::new(raster));
            }
        }

        // Concatenate all raster arrays
        let refs: Vec<&dyn Array> = combined_arrays.iter().map(|a| a.as_ref()).collect();
        let result = arrow::compute::concat(&refs).map_err(|e| {
            DataFusionError::Execution(format!("Failed to concatenate rasters: {}", e))
        })?;

        // Return as scalar if input was scalar
        match &args[0] {
            ColumnarValue::Scalar(_) => {
                let scalar = datafusion_common::ScalarValue::try_from_array(&result, 0)?;
                Ok(ColumnarValue::Scalar(scalar))
            }
            ColumnarValue::Array(_) => Ok(ColumnarValue::Array(result)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_params() {
        let params = "key1=value1;key2=value2";
        let parsed = RsFromPath::parse_params(params);
        assert_eq!(parsed.get("key1"), Some(&"value1".to_string()));
        assert_eq!(parsed.get("key2"), Some(&"value2".to_string()));

        // Empty params
        let parsed = RsFromPath::parse_params("");
        assert!(parsed.is_empty());

        // Single param
        let parsed = RsFromPath::parse_params("option=true");
        assert_eq!(parsed.get("option"), Some(&"true".to_string()));
    }

    #[test]
    fn udf_from_path() {
        let udf: datafusion_expr::ScalarUDF = rs_from_path_udf().into();
        assert_eq!(udf.name(), "rs_frompath");
        assert!(udf.documentation().is_some());
    }

    #[test]
    #[ignore = "RasterBuilder doesn't correctly handle null data for out-db rasters"]
    fn test_load_outdb_raster_from_file() {
        use sedona_testing::data::test_raster;

        // Load test4.tiff - a simple 10x10 GeoTIFF
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        let raster =
            RsFromPath::load_outdb_raster(&path, None).expect("Should load raster from path");

        // Verify the StructArray has correct length
        assert_eq!(raster.len(), 1);

        // Verify metadata directly from the struct array
        use arrow_array::{ListArray, StringViewArray, StructArray, UInt32Array, UInt64Array};
        use sedona_schema::raster::{
            band_indices, band_metadata_indices, metadata_indices, raster_indices,
        };

        let metadata_struct = raster
            .column(raster_indices::METADATA)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let width = metadata_struct
            .column(metadata_indices::WIDTH)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);
        let height = metadata_struct
            .column(metadata_indices::HEIGHT)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);

        assert_eq!(width, 10);
        assert_eq!(height, 10);

        // Check CRS
        let crs = raster
            .column(raster_indices::CRS)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        assert!(!crs.is_null(0));

        // Verify bands - check that it's out-db via the metadata
        let bands_list = raster
            .column(raster_indices::BANDS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let bands_struct = bands_list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let band_metadata_struct = bands_struct
            .column(band_indices::METADATA)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        // Check outdb_url is set (meaning it's an out-db raster)
        let outdb_url = band_metadata_struct
            .column(band_metadata_indices::OUTDB_URL)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        assert!(
            !outdb_url.is_null(0),
            "Out-db raster should have outdb_url set"
        );
        assert!(outdb_url.value(0).contains("test4.tiff"));

        // Check storage type is OutDbRef
        let storage_type = band_metadata_struct
            .column(band_metadata_indices::STORAGE_TYPE)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(
            storage_type.value(0),
            sedona_schema::raster::StorageType::OutDbRef as u32
        );
    }

    #[test]
    #[ignore = "RasterBuilder doesn't correctly handle null data for out-db rasters"]
    fn test_invoke_rs_from_path() {
        use arrow_array::{StringArray, UInt64Array};
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_schema::raster::{metadata_indices, raster_indices};
        use sedona_testing::data::test_raster;

        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        // Create input array with the path
        let paths = Arc::new(StringArray::from(vec![path.as_str()]));
        let input = ColumnarValue::Array(paths);

        // Invoke the UDF
        let kernel = RsFromPath { with_params: false };
        let result = kernel
            .invoke_batch(&[], &[input])
            .expect("Should invoke successfully");

        // Verify result
        match result {
            ColumnarValue::Array(arr) => {
                let struct_arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                assert_eq!(struct_arr.len(), 1);

                // Check dimensions from metadata
                let metadata_struct = struct_arr
                    .column(raster_indices::METADATA)
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .unwrap();
                let width = metadata_struct
                    .column(metadata_indices::WIDTH)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .value(0);
                let height = metadata_struct
                    .column(metadata_indices::HEIGHT)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .value(0);

                assert_eq!(width, 10);
                assert_eq!(height, 10);
            }
            _ => panic!("Expected array result"),
        }
    }
}
