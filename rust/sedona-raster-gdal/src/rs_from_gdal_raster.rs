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

//! RS_FromGDALRaster UDF - Parse binary content using GDAL driver as in-db raster
//!
//! Similar to PostGIS's ST_FromGDALRaster. Parses binary content using GDAL driver
//! and loads it as an in-db raster with all band data stored inline.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, StructArray};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use gdal::spatial_ref::SpatialRef;
use gdal::vsi::{create_mem_file, unlink_mem_file};
use gdal::{Dataset, DatasetOptions, GdalOpenFlags};

use arrow_schema::DataType;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata};
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::{gdal_to_band_data_type, nodata_f64_to_bytes};
use crate::gdal_dataset_provider::configure_thread_local_cache_size;

/// Counter for generating unique VSI memory file names
static VSI_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// RS_FromGDALRaster() scalar UDF implementation
///
/// Parse binary content using GDAL driver and load it as in-db raster
pub fn rs_from_gdal_raster_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_fromgdalraster",
        vec![Arc::new(RsFromGDALRaster)],
        Volatility::Immutable,
        Some(rs_from_gdal_raster_doc()),
    )
}

fn rs_from_gdal_raster_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Parse binary content using GDAL driver and load as in-db raster".to_string(),
        "RS_FromGDALRaster(content: Binary)".to_string(),
    )
    .with_argument("content", "Binary content of a raster file (GeoTiff, etc.)")
    .with_sql_example("SELECT RS_FromGDALRaster(raster_bytes)".to_string())
    .build()
}

/// Kernel implementation for RS_FromGDALRaster
#[derive(Debug)]
pub(crate) struct RsFromGDALRaster;

impl RsFromGDALRaster {
    /// Generate a unique VSI memory file path
    fn generate_vsi_path() -> String {
        let counter = VSI_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let thread_id = std::thread::current().id();
        format!(
            "/vsimem/rs_from_gdal_raster_{:?}_{}.bin",
            thread_id, counter
        )
    }

    /// Read band data as bytes
    fn read_band_data(
        dataset: &Dataset,
        band_idx: usize,
        width: usize,
        height: usize,
        band_type: BandDataType,
    ) -> Result<Vec<u8>> {
        let band = dataset.rasterband(band_idx).map_err(|e| {
            DataFusionError::Execution(format!("Failed to get band {}: {}", band_idx, e))
        })?;

        // Read band data based on type
        let data: Vec<u8> = match band_type {
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

    /// Parse binary content and create an in-db raster
    pub(crate) fn parse_gdal_raster(content: &[u8]) -> Result<StructArray> {
        // Create a temporary VSI memory file
        let vsi_path = Self::generate_vsi_path();
        let content_copy = content.to_vec();

        // Write content to VSI memory file
        create_mem_file(&vsi_path, content_copy).map_err(|e| {
            DataFusionError::Execution(format!("Failed to create VSI memory file: {}", e))
        })?;

        // Open dataset from VSI memory file
        let dataset = Dataset::open_ex(
            &vsi_path,
            DatasetOptions {
                open_flags: GdalOpenFlags::GDAL_OF_RASTER | GdalOpenFlags::GDAL_OF_READONLY,
                ..Default::default()
            },
        )
        .map_err(|e| {
            // Clean up on error
            let _ = unlink_mem_file(&vsi_path);
            DataFusionError::Execution(format!("Failed to open raster from binary: {}", e))
        })?;

        // Get raster dimensions
        let (width, height) = dataset.raster_size();

        // Get geotransform
        let geotransform = dataset.geo_transform().map_err(|e| {
            let _ = unlink_mem_file(&vsi_path);
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
            .map_err(|e| {
                let _ = unlink_mem_file(&vsi_path);
                DataFusionError::Execution(format!("Failed to start raster: {}", e))
            })?;

        // Add bands with in-db data
        let band_count = dataset.raster_count();
        for band_idx in 1..=band_count {
            let band = dataset.rasterband(band_idx).map_err(|e| {
                let _ = unlink_mem_file(&vsi_path);
                DataFusionError::Execution(format!("Failed to get band {}: {}", band_idx, e))
            })?;

            let gdal_type = band.band_type();
            let band_data_type = gdal_to_band_data_type(gdal_type).map_err(|_| {
                let _ = unlink_mem_file(&vsi_path);
                DataFusionError::Execution(format!("Unsupported band data type: {:?}", gdal_type))
            })?;

            // Get nodata value
            let nodata_bytes = band
                .no_data_value()
                .map(|no_data| nodata_f64_to_bytes(no_data, &band_data_type));

            let band_metadata = BandMetadata {
                nodata_value: nodata_bytes,
                storage_type: StorageType::InDb,
                datatype: band_data_type,
                outdb_url: None,
                outdb_band_id: None,
            };

            builder.start_band(band_metadata).map_err(|e| {
                let _ = unlink_mem_file(&vsi_path);
                DataFusionError::Execution(format!("Failed to start band: {}", e))
            })?;

            // Read and store band data
            let band_data = Self::read_band_data(&dataset, band_idx, width, height, band_data_type)
                .inspect_err(|_| {
                    let _ = unlink_mem_file(&vsi_path);
                })?;

            builder.band_data_writer().append_value(&band_data);

            builder.finish_band().map_err(|e| {
                let _ = unlink_mem_file(&vsi_path);
                DataFusionError::Execution(format!("Failed to finish band: {}", e))
            })?;
        }

        // Close dataset before cleaning up VSI file
        drop(dataset);

        // Clean up VSI memory file
        let _ = unlink_mem_file(&vsi_path);

        builder
            .finish_raster()
            .map_err(|e| DataFusionError::Execution(format!("Failed to finish raster: {}", e)))?;

        builder
            .finish()
            .map_err(|e| DataFusionError::Execution(format!("Failed to build raster: {}", e)))
    }
}

impl SedonaScalarKernel for RsFromGDALRaster {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_binary()], RASTER);
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
        // Get the binary content argument
        let content_array = match &args[0] {
            ColumnarValue::Scalar(scalar) => scalar.to_array().map_err(|e| {
                DataFusionError::Execution(format!("Failed to convert scalar to array: {}", e))
            })?,
            ColumnarValue::Array(array) => array.clone(),
        };

        let binary_array = content_array
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected binary array for content argument".to_string())
            })?;

        let len = binary_array.len();

        if len == 0 {
            // Return empty raster array
            let builder = RasterBuilder::new(0);
            let result = builder.finish().map_err(|e| {
                DataFusionError::Execution(format!("Failed to build empty raster: {}", e))
            })?;
            return Ok(ColumnarValue::Array(Arc::new(result)));
        }

        // Process each binary content
        let mut combined_arrays: Vec<ArrayRef> = Vec::with_capacity(len);

        for i in 0..len {
            if binary_array.is_null(i) {
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
                let content = binary_array.value(i);
                let raster = Self::parse_gdal_raster(content)?;
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
    use sedona_raster::traits::RasterRef;

    #[test]
    fn test_generate_vsi_path() {
        let path1 = RsFromGDALRaster::generate_vsi_path();
        let path2 = RsFromGDALRaster::generate_vsi_path();

        assert!(path1.starts_with("/vsimem/rs_from_gdal_raster_"));
        assert!(path2.starts_with("/vsimem/rs_from_gdal_raster_"));
        assert_ne!(path1, path2);
    }

    #[test]
    fn udf_from_gdal_raster() {
        let udf: datafusion_expr::ScalarUDF = rs_from_gdal_raster_udf().into();
        assert_eq!(udf.name(), "rs_fromgdalraster");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn test_parse_geotiff_bytes() {
        use sedona_raster::array::RasterStructArray;
        use sedona_testing::data::test_raster;

        // Read test4.tiff file into bytes
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");
        let content = std::fs::read(&path).expect("Should read file");

        // Parse the GeoTiff bytes into a raster
        let result =
            RsFromGDALRaster::parse_gdal_raster(&content).expect("Should parse GeoTiff bytes");

        // Verify the raster
        let raster_array = RasterStructArray::new(&result);
        assert_eq!(raster_array.len(), 1);

        let raster = raster_array.get(0).expect("Should get raster");
        assert_eq!(raster.metadata().width(), 10);
        assert_eq!(raster.metadata().height(), 10);
        assert_eq!(raster.bands().len(), 1);
        // Check CRS - test4.tiff has EPSG:4326
        assert!(raster.crs().is_some());

        // Verify it's an in-db raster (should have band data, not outdb_url)
        let band = raster.bands().band(1).expect("Should have band 1");
        assert!(
            band.metadata().outdb_url().is_none(),
            "In-db raster should not have outdb_url"
        );
    }

    #[test]
    fn test_invoke_rs_from_gdal_raster() {
        use arrow_array::BinaryArray;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_testing::data::test_raster;

        // Read test file into bytes
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");
        let content = std::fs::read(&path).expect("Should read file");

        // Create binary array with the content
        let binary_arr = Arc::new(BinaryArray::from(vec![content.as_slice()]));
        let input = ColumnarValue::Array(binary_arr);

        // Invoke the UDF
        let kernel = RsFromGDALRaster;
        let result = kernel
            .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
            .expect("Should invoke successfully");

        // Verify result
        match result {
            ColumnarValue::Array(arr) => {
                let struct_arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                let raster_array = sedona_raster::array::RasterStructArray::new(struct_arr);
                assert_eq!(raster_array.len(), 1);
                let raster = raster_array.get(0).expect("Should get raster");
                assert_eq!(raster.metadata().width(), 10);
                assert_eq!(raster.metadata().height(), 10);
            }
            _ => panic!("Expected array result"),
        }
    }
}
