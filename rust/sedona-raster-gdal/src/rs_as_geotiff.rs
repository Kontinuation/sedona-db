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

//! RS_AsGeoTiff UDF - Export raster as GeoTiff binary
//!
//! Returns a binary DataFrame from a Raster DataFrame with multiple overloads:
//! - RS_AsGeoTiff(raster)
//! - RS_AsGeoTiff(raster, tileSize)
//! - RS_AsGeoTiff(raster, compressionType, imageQuality)
//! - RS_AsGeoTiff(raster, compressionType, imageQuality, tileSize)
//! - RS_AsGeoTiff(raster, compressionType, imageQuality, tileWidth, tileHeight)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use arrow_array::{Array, Float64Array, StringArray, StructArray, UInt32Array};
use arrow_schema::DataType;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use gdal::raster::RasterCreationOptions;
use gdal::vsi::{get_vsi_mem_file_bytes_owned, unlink_mem_file};
use gdal::DriverManager;

use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::{RasterRefImpl, RasterStructArray};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

// Use thread-local provider to create GDAL datasets from `RasterRef`.
use crate::gdal_dataset_provider::configure_thread_local_cache_size;

/// Counter for generating unique VSI memory file names
static VSI_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Compression types supported for GeoTiff output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None,
    PackBits,
    Deflate,
    Huffman,
    Lzw,
    Jpeg,
}

impl CompressionType {
    /// Parse compression type from string (case-insensitive)
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(CompressionType::None),
            "packbits" => Some(CompressionType::PackBits),
            "deflate" => Some(CompressionType::Deflate),
            "huffman" => Some(CompressionType::Huffman),
            "lzw" => Some(CompressionType::Lzw),
            "jpeg" => Some(CompressionType::Jpeg),
            _ => None,
        }
    }

    /// Get GDAL compression option value
    pub fn gdal_value(&self) -> &'static str {
        match self {
            CompressionType::None => "NONE",
            CompressionType::PackBits => "PACKBITS",
            CompressionType::Deflate => "DEFLATE",
            CompressionType::Huffman => "CCITTRLE",
            CompressionType::Lzw => "LZW",
            CompressionType::Jpeg => "JPEG",
        }
    }
}

/// RS_AsGeoTiff() scalar UDF implementation
///
/// Returns a binary DataFrame from a Raster DataFrame
pub fn rs_as_geotiff_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_asgeotiff",
        vec![
            Arc::new(RsAsGeoTiff::new(Variant::Basic)), // RS_AsGeoTiff(raster)
            Arc::new(RsAsGeoTiff::new(Variant::WithTileSize)), // RS_AsGeoTiff(raster, tileSize)
            Arc::new(RsAsGeoTiff::new(Variant::WithCompressionQuality)), // RS_AsGeoTiff(raster, compression, quality)
            Arc::new(RsAsGeoTiff::new(Variant::WithCompressionQualityTileSize)), // RS_AsGeoTiff(raster, compression, quality, tileSize)
            Arc::new(RsAsGeoTiff::new(Variant::WithCompressionQualityTileWH)), // RS_AsGeoTiff(raster, compression, quality, tileWidth, tileHeight)
        ],
        Volatility::Immutable,
        Some(rs_as_geotiff_doc()),
    )
}

fn rs_as_geotiff_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns a binary (GeoTiff) representation of a raster".to_string(),
        "RS_AsGeoTiff(raster: Raster[, compressionType: String, imageQuality: Double[, tileSize: Int | tileWidth: Int, tileHeight: Int]])".to_string(),
    )
    .with_argument("raster", "Input raster")
    .with_argument("compressionType", "Optional compression type: None, PackBits, Deflate, Huffman, LZW, JPEG")
    .with_argument("imageQuality", "Optional image quality for JPEG compression (0.0-1.0)")
    .with_argument("tileSize", "Optional tile size (applies to both width and height)")
    .with_argument("tileWidth", "Optional tile width")
    .with_argument("tileHeight", "Optional tile height")
    .with_sql_example(
        "SELECT RS_AsGeoTiff(raster)\nSELECT RS_AsGeoTiff(raster, 256)\nSELECT RS_AsGeoTiff(raster, 'DEFLATE', 0.9)\nSELECT RS_AsGeoTiff(raster, 'JPEG', 0.85, 512)\nSELECT RS_AsGeoTiff(raster, 'LZW', 0.9, 256, 256)".to_string(),
    )
    .build()
}

/// Variants for different overloads
#[derive(Debug, Clone, Copy)]
enum Variant {
    Basic,                          // (raster)
    WithTileSize,                   // (raster, tileSize)
    WithCompressionQuality,         // (raster, compression, quality)
    WithCompressionQualityTileSize, // (raster, compression, quality, tileSize)
    WithCompressionQualityTileWH,   // (raster, compression, quality, tileWidth, tileHeight)
}

/// Kernel implementation for RS_AsGeoTiff
#[derive(Debug)]
struct RsAsGeoTiff {
    variant: Variant,
}

impl RsAsGeoTiff {
    fn new(variant: Variant) -> Self {
        Self { variant }
    }

    /// Generate a unique VSI memory file path
    fn generate_vsi_path() -> String {
        let counter = VSI_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let thread_id = std::thread::current().id();
        format!("/vsimem/rs_as_geotiff_{:?}_{}.tif", thread_id, counter)
    }

    /// Convert raster to GeoTiff bytes
    fn raster_to_geotiff(
        raster: &RasterRefImpl,
        compression: Option<CompressionType>,
        quality: Option<f64>,
        tile_width: Option<u32>,
        tile_height: Option<u32>,
    ) -> Result<Vec<u8>> {
        // Create GDAL dataset from raster using the thread-local provider
        let provider = crate::gdal_dataset_provider::thread_local_provider().map_err(|e| {
            DataFusionError::Execution(format!("Failed to init GDAL provider: {}", e))
        })?;
        let raster_ds = provider.raster_ref_to_gdal(raster).map_err(|e| {
            DataFusionError::Execution(format!("Failed to create GDAL dataset: {}", e))
        })?;
        let source_dataset = raster_ds.as_dataset();

        // Get GeoTiff driver
        let driver = DriverManager::get_driver_by_name("GTiff").map_err(|e| {
            DataFusionError::Execution(format!("Failed to get GTiff driver: {}", e))
        })?;

        // Build creation options as string list
        let mut options_list: Vec<String> = Vec::new();

        // Add compression option
        if let Some(comp) = compression {
            options_list.push(format!("COMPRESS={}", comp.gdal_value()));

            // Add quality for JPEG
            if comp == CompressionType::Jpeg {
                if let Some(q) = quality {
                    // JPEG quality is 1-100, we receive 0.0-1.0
                    let jpeg_quality = (q * 100.0).round() as i32;
                    options_list.push(format!("JPEG_QUALITY={}", jpeg_quality.clamp(1, 100)));
                }
            }

            // Add predictor for Deflate/LZW (improves compression)
            if comp == CompressionType::Deflate || comp == CompressionType::Lzw {
                options_list.push("PREDICTOR=2".to_string());
            }
        }

        // Add tiling options
        if let (Some(tw), Some(th)) = (tile_width, tile_height) {
            options_list.push("TILED=YES".to_string());
            options_list.push(format!("BLOCKXSIZE={}", tw));
            options_list.push(format!("BLOCKYSIZE={}", th));
        }

        // Convert to RasterCreationOptions
        let options = RasterCreationOptions::from_iter(options_list.iter().map(|s| s.as_str()));

        // Generate VSI path for output
        let vsi_path = Self::generate_vsi_path();

        // Create copy to VSI memory file
        let _output_dataset = source_dataset
            .create_copy(&driver, &vsi_path, &options)
            .map_err(|e| DataFusionError::Execution(format!("Failed to create GeoTiff: {}", e)))?;

        // Close the output dataset to flush data
        drop(_output_dataset);

        // Read bytes from VSI memory file and clean up
        let bytes = get_vsi_mem_file_bytes_owned(&vsi_path).map_err(|e| {
            let _ = unlink_mem_file(&vsi_path);
            DataFusionError::Execution(format!("Failed to read GeoTiff bytes: {}", e))
        })?;

        // Clean up VSI file
        let _ = unlink_mem_file(&vsi_path);

        Ok(bytes)
    }

    /// Helper to extract scalar value
    fn get_scalar_value<T>(col: &ColumnarValue, idx: usize) -> Result<T>
    where
        T: TryFromScalar,
    {
        T::try_from_columnar(col, idx)
    }
}

trait TryFromScalar: Sized {
    fn try_from_columnar(col: &ColumnarValue, idx: usize) -> Result<Self>;
}

impl TryFromScalar for String {
    fn try_from_columnar(col: &ColumnarValue, idx: usize) -> Result<Self> {
        match col {
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(s))) => Ok(s.clone()),
            ColumnarValue::Scalar(ScalarValue::Utf8(None)) => {
                Err(DataFusionError::Execution("Null string value".to_string()))
            }
            ColumnarValue::Array(arr) => {
                let string_arr = arr.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                    DataFusionError::Execution("Expected string array".to_string())
                })?;
                if string_arr.is_null(idx) {
                    Err(DataFusionError::Execution("Null string value".to_string()))
                } else {
                    Ok(string_arr.value(idx).to_string())
                }
            }
            _ => Err(DataFusionError::Execution(
                "Expected string value".to_string(),
            )),
        }
    }
}

impl TryFromScalar for f64 {
    fn try_from_columnar(col: &ColumnarValue, idx: usize) -> Result<Self> {
        match col {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(f))) => Ok(*f),
            ColumnarValue::Scalar(ScalarValue::Float64(None)) => {
                Err(DataFusionError::Execution("Null float value".to_string()))
            }
            ColumnarValue::Array(arr) => {
                let float_arr = arr.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                    DataFusionError::Execution("Expected float64 array".to_string())
                })?;
                if float_arr.is_null(idx) {
                    Err(DataFusionError::Execution("Null float value".to_string()))
                } else {
                    Ok(float_arr.value(idx))
                }
            }
            _ => Err(DataFusionError::Execution(
                "Expected float64 value".to_string(),
            )),
        }
    }
}

impl TryFromScalar for u32 {
    fn try_from_columnar(col: &ColumnarValue, idx: usize) -> Result<Self> {
        match col {
            ColumnarValue::Scalar(ScalarValue::UInt32(Some(v))) => Ok(*v),
            ColumnarValue::Scalar(ScalarValue::UInt32(None)) => {
                Err(DataFusionError::Execution("Null uint32 value".to_string()))
            }
            ColumnarValue::Scalar(ScalarValue::Int32(Some(v))) => Ok(*v as u32),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(v))) => Ok(*v as u32),
            ColumnarValue::Array(arr) => {
                let uint_arr = arr.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| {
                    DataFusionError::Execution("Expected uint32 array".to_string())
                })?;
                if uint_arr.is_null(idx) {
                    Err(DataFusionError::Execution("Null uint32 value".to_string()))
                } else {
                    Ok(uint_arr.value(idx))
                }
            }
            _ => Err(DataFusionError::Execution(
                "Expected uint32 value".to_string(),
            )),
        }
    }
}

impl SedonaScalarKernel for RsAsGeoTiff {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = match self.variant {
            Variant::Basic => vec![ArgMatcher::is_raster()],
            Variant::WithTileSize => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(), // tileSize
            ],
            Variant::WithCompressionQuality => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),  // compressionType
                ArgMatcher::is_numeric(), // imageQuality
            ],
            Variant::WithCompressionQualityTileSize => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),  // compressionType
                ArgMatcher::is_numeric(), // imageQuality
                ArgMatcher::is_integer(), // tileSize
            ],
            Variant::WithCompressionQualityTileWH => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),  // compressionType
                ArgMatcher::is_numeric(), // imageQuality
                ArgMatcher::is_integer(), // tileWidth
                ArgMatcher::is_integer(), // tileHeight
            ],
        };

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(DataType::Binary));
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
        // Get the raster argument
        let raster_col = &args[0];

        // Determine number of iterations
        let num_iterations = match raster_col {
            ColumnarValue::Array(arr) => arr.len(),
            ColumnarValue::Scalar(_) => 1,
        };

        // Build output binary array
        let mut builder = BinaryBuilder::with_capacity(num_iterations, num_iterations * 1024);

        // Get raster array
        let raster_array: RasterStructArray = match raster_col {
            ColumnarValue::Array(arr) => {
                let struct_arr = arr.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
                    DataFusionError::Execution("Expected StructArray for raster".to_string())
                })?;
                RasterStructArray::new(struct_arr)
            }
            ColumnarValue::Scalar(ScalarValue::Struct(arc_struct)) => {
                RasterStructArray::new(arc_struct.as_ref())
            }
            ColumnarValue::Scalar(ScalarValue::Null) => {
                builder.append_null();
                let result = builder.finish();
                return Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                    &result, 0,
                )?));
            }
            _ => {
                return Err(DataFusionError::Execution(
                    "Expected raster value".to_string(),
                ))
            }
        };

        for i in 0..num_iterations {
            if raster_array.is_null(i) {
                builder.append_null();
                continue;
            }

            let raster = raster_array.get(i).map_err(|e| {
                DataFusionError::Execution(format!("Failed to get raster at index {}: {}", i, e))
            })?;

            // Parse variant-specific arguments
            let (compression, quality, tile_width, tile_height) = match self.variant {
                Variant::Basic => (None, None, None, None),
                Variant::WithTileSize => {
                    let tile_size: u32 = Self::get_scalar_value(&args[1], i)?;
                    (None, None, Some(tile_size), Some(tile_size))
                }
                Variant::WithCompressionQuality => {
                    let comp_str: String = Self::get_scalar_value(&args[1], i)?;
                    let compression = CompressionType::parse(&comp_str).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Unknown compression type: {}. Valid values: None, PackBits, Deflate, Huffman, LZW, JPEG",
                            comp_str
                        ))
                    })?;
                    let quality: f64 = Self::get_scalar_value(&args[2], i)?;
                    (Some(compression), Some(quality), None, None)
                }
                Variant::WithCompressionQualityTileSize => {
                    let comp_str: String = Self::get_scalar_value(&args[1], i)?;
                    let compression = CompressionType::parse(&comp_str).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Unknown compression type: {}. Valid values: None, PackBits, Deflate, Huffman, LZW, JPEG",
                            comp_str
                        ))
                    })?;
                    let quality: f64 = Self::get_scalar_value(&args[2], i)?;
                    let tile_size: u32 = Self::get_scalar_value(&args[3], i)?;
                    (
                        Some(compression),
                        Some(quality),
                        Some(tile_size),
                        Some(tile_size),
                    )
                }
                Variant::WithCompressionQualityTileWH => {
                    let comp_str: String = Self::get_scalar_value(&args[1], i)?;
                    let compression = CompressionType::parse(&comp_str).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Unknown compression type: {}. Valid values: None, PackBits, Deflate, Huffman, LZW, JPEG",
                            comp_str
                        ))
                    })?;
                    let quality: f64 = Self::get_scalar_value(&args[2], i)?;
                    let tile_width: u32 = Self::get_scalar_value(&args[3], i)?;
                    let tile_height: u32 = Self::get_scalar_value(&args[4], i)?;
                    (
                        Some(compression),
                        Some(quality),
                        Some(tile_width),
                        Some(tile_height),
                    )
                }
            };

            // Convert raster to GeoTiff
            let bytes =
                Self::raster_to_geotiff(&raster, compression, quality, tile_width, tile_height)?;
            builder.append_value(&bytes);
        }

        let result = builder.finish();

        // Return as scalar if input was scalar
        match raster_col {
            ColumnarValue::Scalar(_) => {
                let scalar = ScalarValue::try_from_array(&result, 0)?;
                Ok(ColumnarValue::Scalar(scalar))
            }
            ColumnarValue::Array(_) => Ok(ColumnarValue::Array(Arc::new(result))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configure_gdal_shim_from_current_process;
    use sedona_raster::traits::RasterRef;

    #[test]
    fn test_compression_type_parse() {
        assert_eq!(CompressionType::parse("none"), Some(CompressionType::None));
        assert_eq!(CompressionType::parse("NONE"), Some(CompressionType::None));
        assert_eq!(
            CompressionType::parse("deflate"),
            Some(CompressionType::Deflate)
        );
        assert_eq!(
            CompressionType::parse("DEFLATE"),
            Some(CompressionType::Deflate)
        );
        assert_eq!(CompressionType::parse("lzw"), Some(CompressionType::Lzw));
        assert_eq!(CompressionType::parse("jpeg"), Some(CompressionType::Jpeg));
        assert_eq!(CompressionType::parse("invalid"), None);
    }

    #[test]
    fn test_generate_vsi_path() {
        let path1 = RsAsGeoTiff::generate_vsi_path();
        let path2 = RsAsGeoTiff::generate_vsi_path();

        assert!(path1.starts_with("/vsimem/rs_as_geotiff_"));
        assert!(path1.ends_with(".tif"));
        assert!(path2.starts_with("/vsimem/rs_as_geotiff_"));
        assert_ne!(path1, path2);
    }

    #[test]
    fn udf_as_geotiff() {
        let udf: datafusion_expr::ScalarUDF = rs_as_geotiff_udf().into();
        assert_eq!(udf.name(), "rs_asgeotiff");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn test_roundtrip_geotiff() {
        configure_gdal_shim_from_current_process().unwrap();
        use crate::rs_from_gdal_raster::RsFromGDALRaster;
        use sedona_raster::array::RasterStructArray;
        use sedona_testing::data::test_raster;

        // Load test4.tiff as in-db raster
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");
        let content = std::fs::read(&path).expect("Should read file");
        let raster_arr =
            RsFromGDALRaster::parse_gdal_raster(&content).expect("Should parse GeoTiff");

        // Get the raster using RasterStructArray
        let raster_array = RasterStructArray::new(&raster_arr);
        assert_eq!(raster_array.len(), 1);
        let raster = raster_array.get(0).expect("Should get raster");

        // Convert to GeoTiff bytes
        let geotiff_bytes = RsAsGeoTiff::raster_to_geotiff(&raster, None, None, None, None)
            .expect("Should convert to GeoTiff");

        // Verify we got valid GeoTiff data (check magic bytes)
        assert!(geotiff_bytes.len() > 4, "GeoTiff should have content");
        // GeoTiff starts with either II (little-endian) or MM (big-endian)
        assert!(
            &geotiff_bytes[0..2] == b"II" || &geotiff_bytes[0..2] == b"MM",
            "Should be valid TIFF header"
        );

        // Parse it back and verify dimensions
        let roundtrip_arr = RsFromGDALRaster::parse_gdal_raster(&geotiff_bytes)
            .expect("Should parse roundtrip GeoTiff");
        let roundtrip_array = RasterStructArray::new(&roundtrip_arr);
        let roundtrip_raster = roundtrip_array.get(0).expect("Should get roundtrip raster");

        assert_eq!(
            roundtrip_raster.metadata().width(),
            raster.metadata().width()
        );
        assert_eq!(
            roundtrip_raster.metadata().height(),
            raster.metadata().height()
        );
        assert_eq!(roundtrip_raster.bands().len(), raster.bands().len());
    }

    #[test]
    fn test_geotiff_with_compression() {
        configure_gdal_shim_from_current_process().unwrap();
        use crate::rs_from_gdal_raster::RsFromGDALRaster;
        use sedona_raster::array::RasterStructArray;
        use sedona_testing::data::test_raster;

        // Load test raster
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");
        let content = std::fs::read(&path).expect("Should read file");
        let raster_arr =
            RsFromGDALRaster::parse_gdal_raster(&content).expect("Should parse GeoTiff");

        let raster_array = RasterStructArray::new(&raster_arr);
        let raster = raster_array.get(0).expect("Should get raster");

        // Test with LZW compression
        let lzw_bytes = RsAsGeoTiff::raster_to_geotiff(
            &raster,
            Some(CompressionType::Lzw),
            Some(75.0),
            None,
            None,
        )
        .expect("Should convert with LZW");
        assert!(
            !lzw_bytes.is_empty(),
            "LZW compressed GeoTiff should have content"
        );

        // Test with DEFLATE compression
        let deflate_bytes = RsAsGeoTiff::raster_to_geotiff(
            &raster,
            Some(CompressionType::Deflate),
            Some(6.0),
            None,
            None,
        )
        .expect("Should convert with DEFLATE");
        assert!(
            !deflate_bytes.is_empty(),
            "DEFLATE compressed GeoTiff should have content"
        );

        // Both should be valid TIFFs
        assert!(&lzw_bytes[0..2] == b"II" || &lzw_bytes[0..2] == b"MM");
        assert!(&deflate_bytes[0..2] == b"II" || &deflate_bytes[0..2] == b"MM");
    }
}
