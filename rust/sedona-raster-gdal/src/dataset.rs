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

//! GDAL Dataset conversion utilities for raster data.
//!
//! This module provides functions to convert RasterRef to GDALDataset:
//! - For in-db rasters: Creates a GDALDataset backed by the MEM driver, with zero-copy
//!   access to the band data stored in the Arrow array.
//! - For out-db rasters: Creates a GDALDataset as a VRT (Virtual Raster) that references
//!   the external data sources.

use std::ffi::{c_void, CString};
use std::marker::PhantomData;
use std::ptr::null_mut;

use gdal::errors::{GdalError, Result};
use gdal::raster::GdalDataType;
use gdal::vrt::VrtDataset;
use gdal::{Dataset, DatasetOptions, GdalOpenFlags};
use gdal_sys::{
    CPLErr, GDALAddBand, GDALClose, GDALCreate, GDALDatasetH, GDALDriverH, GDALGetDriverByName,
    GDALGetRasterBand, GDALSetGeoTransform, GDALSetProjection, GDALSetRasterNoDataValue,
};

use sedona_raster::traits::RasterRef;
use sedona_schema::raster::{BandDataType, StorageType};

/// Converts a BandDataType to the corresponding GDAL data type.
pub fn band_data_type_to_gdal(band_type: &BandDataType) -> GdalDataType {
    match band_type {
        BandDataType::UInt8 => GdalDataType::UInt8,
        BandDataType::UInt16 => GdalDataType::UInt16,
        BandDataType::Int16 => GdalDataType::Int16,
        BandDataType::UInt32 => GdalDataType::UInt32,
        BandDataType::Int32 => GdalDataType::Int32,
        BandDataType::Float32 => GdalDataType::Float32,
        BandDataType::Float64 => GdalDataType::Float64,
    }
}

/// Converts a GDAL data type to the corresponding BandDataType.
pub fn gdal_to_band_data_type(gdal_type: GdalDataType) -> Option<BandDataType> {
    match gdal_type {
        GdalDataType::UInt8 => Some(BandDataType::UInt8),
        GdalDataType::UInt16 => Some(BandDataType::UInt16),
        GdalDataType::Int16 => Some(BandDataType::Int16),
        GdalDataType::UInt32 => Some(BandDataType::UInt32),
        GdalDataType::Int32 => Some(BandDataType::Int32),
        GdalDataType::Float32 => Some(BandDataType::Float32),
        GdalDataType::Float64 => Some(BandDataType::Float64),
        _ => None, // CInt16, CInt32, CFloat32, CFloat64, Unknown
    }
}

/// Returns the byte size of a GDAL data type.
pub fn gdal_type_byte_size(gdal_type: GdalDataType) -> usize {
    match gdal_type {
        GdalDataType::UInt8 => 1,
        GdalDataType::UInt16 | GdalDataType::Int16 => 2,
        GdalDataType::UInt32 | GdalDataType::Int32 | GdalDataType::Float32 => 4,
        GdalDataType::Float64 => 8,
        _ => 0, // Complex types not supported
    }
}

/// Interprets nodata bytes according to the band data type and returns as f64.
///
/// Returns None if the nodata_bytes is None or has incorrect length.
pub fn nodata_bytes_to_f64(nodata_bytes: Option<&[u8]>, band_type: &BandDataType) -> Option<f64> {
    let bytes = nodata_bytes?;

    match band_type {
        BandDataType::UInt8 => {
            if bytes.len() == 1 {
                Some(bytes[0] as f64)
            } else {
                None
            }
        }
        BandDataType::UInt16 => {
            if bytes.len() == 2 {
                Some(u16::from_le_bytes([bytes[0], bytes[1]]) as f64)
            } else {
                None
            }
        }
        BandDataType::Int16 => {
            if bytes.len() == 2 {
                Some(i16::from_le_bytes([bytes[0], bytes[1]]) as f64)
            } else {
                None
            }
        }
        BandDataType::UInt32 => {
            if bytes.len() == 4 {
                Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
            } else {
                None
            }
        }
        BandDataType::Int32 => {
            if bytes.len() == 4 {
                Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
            } else {
                None
            }
        }
        BandDataType::Float32 => {
            if bytes.len() == 4 {
                Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
            } else {
                None
            }
        }
        BandDataType::Float64 => {
            if bytes.len() == 8 {
                Some(f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]))
            } else {
                None
            }
        }
    }
}

/// Converts a nodata f64 value to bytes according to the band data type.
pub fn nodata_f64_to_bytes(nodata: f64, band_type: &BandDataType) -> Vec<u8> {
    match band_type {
        BandDataType::UInt8 => vec![nodata as u8],
        BandDataType::UInt16 => (nodata as u16).to_le_bytes().to_vec(),
        BandDataType::Int16 => (nodata as i16).to_le_bytes().to_vec(),
        BandDataType::UInt32 => (nodata as u32).to_le_bytes().to_vec(),
        BandDataType::Int32 => (nodata as i32).to_le_bytes().to_vec(),
        BandDataType::Float32 => (nodata as f32).to_le_bytes().to_vec(),
        BandDataType::Float64 => nodata.to_le_bytes().to_vec(),
    }
}

/// A wrapper around a GDAL MEM dataset that provides zero-copy access to raster data.
///
/// This struct holds a reference to the original RasterRef to ensure the underlying
/// Arrow data stays valid for the lifetime of the GDAL dataset. The GDAL MEM driver
/// directly references the memory addresses of the band data, so we must ensure
/// the RasterRef outlives this dataset.
///
/// # Lifetime
/// The `'a` lifetime parameter ensures that this dataset cannot outlive the RasterRef
/// it was created from.
pub struct RasterMemDataset<'a> {
    c_dataset: GDALDatasetH,
    _phantom: PhantomData<&'a ()>,
}

impl<'a> Drop for RasterMemDataset<'a> {
    fn drop(&mut self) {
        unsafe {
            GDALClose(self.c_dataset);
        }
    }
}

impl<'a> RasterMemDataset<'a> {
    /// Returns the raw GDAL dataset handle.
    ///
    /// # Safety
    /// The returned handle is only valid for the lifetime of this struct.
    /// Do not close or transfer ownership of the handle.
    pub unsafe fn c_dataset(&self) -> GDALDatasetH {
        self.c_dataset
    }

    /// Creates a GDAL Dataset (read-only) from this wrapper.
    ///
    /// # Safety
    /// The returned Dataset borrows from this wrapper and must not outlive it.
    /// The caller must ensure the Dataset is dropped before this wrapper.
    pub unsafe fn as_dataset(&self) -> Dataset {
        Dataset::from_c_dataset(self.c_dataset)
    }

    /// Consumes this wrapper and returns a GDAL Dataset, transferring ownership.
    ///
    /// This is the preferred method when you need to return or store the Dataset
    /// separately from this wrapper. The wrapper is consumed without calling its
    /// destructor, so the Dataset becomes solely responsible for closing the handle.
    ///
    /// # Safety
    /// The returned Dataset internally references the original RasterRef's memory.
    /// The caller must ensure the RasterRef outlives the returned Dataset.
    pub unsafe fn into_dataset(self) -> Dataset {
        let dataset = Dataset::from_c_dataset(self.c_dataset);
        std::mem::forget(self);
        dataset
    }
}

/// Creates a GDAL MEM dataset from an in-db raster with zero-copy band data access.
///
/// This function creates a GDAL dataset backed by the MEM driver that directly
/// references the band data stored in the Arrow array. No data copying occurs -
/// the GDAL bands point to the same memory as the Arrow array.
///
/// # Arguments
/// * `raster` - The RasterRef containing the in-db raster data
///
/// # Returns
/// A `RasterMemDataset` that provides access to the GDAL dataset. The returned
/// dataset has the same lifetime as the input RasterRef, ensuring the underlying
/// data remains valid.
///
/// # Errors
/// Returns an error if:
/// - Any band uses OutDb storage (use `raster_to_vrt_dataset` instead)
/// - GDAL driver operations fail
/// - Memory allocation fails
///
/// # Example
/// ```ignore
/// let raster_array = RasterStructArray::new(&struct_array);
/// let raster = raster_array.get(0)?;
/// let mem_dataset = raster_to_mem_dataset(&raster)?;
/// // Use the dataset...
/// // mem_dataset is automatically cleaned up when dropped
/// ```
pub fn raster_to_mem_dataset<'a>(raster: &'a dyn RasterRef) -> Result<RasterMemDataset<'a>> {
    let metadata = raster.metadata();
    let bands = raster.bands();

    // Validate that all bands are InDb storage
    for i in 1..=bands.len() {
        let band = bands
            .band(i)
            .map_err(|e| GdalError::BadArgument(format!("Failed to access band {}: {}", i, e)))?;
        if band.metadata().storage_type() != StorageType::InDb {
            return Err(GdalError::BadArgument(format!(
                "Band {} uses OutDb storage; use raster_to_vrt_dataset instead",
                i
            )));
        }
    }

    let width = metadata.width() as usize;
    let height = metadata.height() as usize;

    // Get MEM driver
    let driver: GDALDriverH = unsafe {
        let driver_name = CString::new("MEM").unwrap();
        GDALGetDriverByName(driver_name.as_ptr())
    };

    if driver.is_null() {
        return Err(GdalError::NullPointer {
            method_name: "GDALGetDriverByName",
            msg: "Could not load MEM GDAL driver".to_string(),
        });
    }

    // Create empty dataset (0 bands initially, we'll add them with DATAPOINTER)
    let c_dataset: GDALDatasetH = unsafe {
        let empty_name = CString::new("").unwrap();
        GDALCreate(
            driver,
            empty_name.as_ptr(),
            width as i32,
            height as i32,
            0, // 0 bands initially
            gdal_sys::GDALDataType::GDT_Byte,
            null_mut(),
        )
    };

    if c_dataset.is_null() {
        return Err(GdalError::NullPointer {
            method_name: "GDALCreate",
            msg: "Could not create GDALDataset".to_string(),
        });
    }

    // Set geotransform
    // GDAL geotransform: [origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height]
    let mut geotransform = [
        metadata.upper_left_x(),
        metadata.scale_x(),
        metadata.skew_x(),
        metadata.upper_left_y(),
        metadata.skew_y(),
        metadata.scale_y(),
    ];

    let rv = unsafe { GDALSetGeoTransform(c_dataset, geotransform.as_mut_ptr()) };
    if rv != CPLErr::CE_None {
        unsafe {
            GDALClose(c_dataset);
        }
        return Err(GdalError::CplError {
            class: rv,
            number: 0,
            msg: "Could not set geotransform".to_string(),
        });
    }

    // Set projection/CRS if available
    if let Some(crs) = raster.crs() {
        let crs_cstring = CString::new(crs)
            .map_err(|_| GdalError::BadArgument("CRS string contains null byte".to_string()))?;
        let rv = unsafe { GDALSetProjection(c_dataset, crs_cstring.as_ptr()) };
        if rv != CPLErr::CE_None {
            unsafe {
                GDALClose(c_dataset);
            }
            return Err(GdalError::CplError {
                class: rv,
                number: 0,
                msg: "Could not set projection".to_string(),
            });
        }
    }

    // Add bands with DATAPOINTER option (zero-copy)
    for i in 1..=bands.len() {
        let band = bands.band(i).map_err(|e| {
            unsafe {
                GDALClose(c_dataset);
            }
            GdalError::BadArgument(format!("Failed to access band {}: {}", i, e))
        })?;

        let band_metadata = band.metadata();
        let band_type = band_metadata.data_type();
        let gdal_type = band_data_type_to_gdal(&band_type);

        // Get pointer to band data
        let band_data = band.data();
        let data_ptr = band_data.as_ptr() as *const c_void;

        // Format the data pointer as a hex string for GDAL
        let datapointer_option = format!("DATAPOINTER={:p}", data_ptr);
        let datapointer_cstring = CString::new(datapointer_option).unwrap();

        // Create options array for GDALAddBand
        let mut options_ptrs = vec![datapointer_cstring.as_ptr() as *mut i8, null_mut()];

        let rv = unsafe { GDALAddBand(c_dataset, gdal_type as u32, options_ptrs.as_mut_ptr()) };

        if rv != CPLErr::CE_None {
            unsafe {
                GDALClose(c_dataset);
            }
            return Err(GdalError::CplError {
                class: rv,
                number: 0,
                msg: format!("Could not add band {}", i),
            });
        }

        // Set nodata value if present
        if let Some(nodata) = nodata_bytes_to_f64(band_metadata.nodata_value(), &band_type) {
            let raster_band = unsafe { GDALGetRasterBand(c_dataset, i as i32) };
            if !raster_band.is_null() {
                unsafe {
                    GDALSetRasterNoDataValue(raster_band, nodata);
                }
            }
        }
    }

    Ok(RasterMemDataset {
        c_dataset,
        _phantom: PhantomData,
    })
}

/// A wrapper around a GDAL VRT dataset for out-db rasters.
///
/// This struct wraps a VRT dataset that references external data sources.
/// Unlike `RasterMemDataset`, this struct owns the GDAL dataset and does not
/// require the original RasterRef to stay alive (the VRT references external files).
///
/// It also keeps references to source datasets open to ensure they remain valid
/// for the lifetime of the VRT.
pub struct RasterVrtDataset {
    vrt: VrtDataset,
    /// Source datasets that must remain open while the VRT is in use
    _source_datasets: Vec<Dataset>,
}

impl RasterVrtDataset {
    /// Returns the underlying VrtDataset.
    pub fn vrt(&self) -> &VrtDataset {
        &self.vrt
    }

    /// Returns the raw GDAL dataset handle.
    ///
    /// # Safety
    /// The returned handle is only valid for the lifetime of this struct.
    /// Do not close or transfer ownership of the handle.
    pub unsafe fn c_dataset(&self) -> GDALDatasetH {
        self.vrt.c_dataset()
    }

    /// Returns a reference to the underlying GDAL `Dataset`.
    pub fn as_dataset(&self) -> &Dataset {
        self.vrt.as_ref()
    }

    /// Consumes this wrapper and returns a GDAL Dataset, transferring ownership.
    ///
    /// This is the preferred method when you need to return or store the Dataset
    /// separately from this wrapper. The wrapper is consumed without calling its
    /// destructor, so the Dataset becomes solely responsible for closing the handle.
    ///
    /// Note: This method leaks the source datasets that were kept alive by this wrapper.
    /// This is acceptable for short-lived operations but may cause resource leaks in
    /// long-running processes.
    pub unsafe fn into_dataset(self) -> Dataset {
        let RasterVrtDataset {
            vrt,
            _source_datasets,
        } = self;

        // `VrtDataset::as_dataset` transfers ownership of the underlying GDAL handle.
        // Keep the source datasets alive by intentionally leaking them.
        let dataset = vrt.as_dataset();
        std::mem::forget(_source_datasets);
        dataset
    }
}

/// Creates a GDAL VRT dataset from an out-db (or mixed) raster.
///
/// This function creates a VRT (Virtual Raster) dataset that references external
/// data sources for out-db bands. For multi-band rasters where bands may come from
/// different external files, this function builds a composite VRT.
///
/// # Arguments
/// * `raster` - The RasterRef containing the raster metadata and band references
///
/// # Returns
/// A `RasterVrtDataset` that provides access to the GDAL VRT dataset.
///
/// # Errors
/// Returns an error if:
/// - Out-db bands don't have valid URL references
/// - GDAL driver operations fail
/// - External data sources cannot be accessed
///
/// # Implementation Notes
/// This function uses the GDAL VRT safe wrapper API which internally calls
/// the VRT C API (like PostGIS does) instead of constructing VRT XML. It:
/// 1. Creates a VRT dataset with VrtDataset::create
/// 2. Opens each external source file
/// 3. Adds VRT bands with add_simple_source references to the external bands
pub fn raster_to_vrt_dataset(raster: &dyn RasterRef) -> Result<RasterVrtDataset> {
    let metadata = raster.metadata();
    let bands = raster.bands();

    if bands.is_empty() {
        return Err(GdalError::BadArgument(
            "Cannot create VRT dataset from raster with no bands".to_string(),
        ));
    }

    let width = metadata.width() as i32;
    let height = metadata.height() as i32;

    // Create VRT dataset using the safe wrapper
    let mut vrt = VrtDataset::create(width as usize, height as usize)?;

    // Set geotransform
    let geotransform = [
        metadata.upper_left_x(),
        metadata.scale_x(),
        metadata.skew_x(),
        metadata.upper_left_y(),
        metadata.skew_y(),
        metadata.scale_y(),
    ];
    vrt.set_geo_transform(&geotransform)?;

    // Set projection/CRS if available
    if let Some(crs) = raster.crs() {
        vrt.set_projection(crs)?;
    }

    // Keep source datasets alive for the lifetime of the VRT
    let mut source_datasets: Vec<Dataset> = Vec::new();

    // Add bands
    for i in 1..=bands.len() {
        let band = bands
            .band(i)
            .map_err(|e| GdalError::BadArgument(format!("Failed to access band {}: {}", i, e)))?;

        let band_metadata = band.metadata();
        let band_type = band_metadata.data_type();
        let gdal_type = band_data_type_to_gdal(&band_type);

        match band_metadata.storage_type() {
            StorageType::OutDbRef => {
                // Get the URL for the external file
                let url = band_metadata.outdb_url().ok_or_else(|| {
                    GdalError::BadArgument(format!("Band {} is OutDbRef but has no URL", i))
                })?;

                // Band ID in the external file (1-based)
                let source_band_num = band_metadata.outdb_band_id().unwrap_or(1) as usize;

                // Open the source dataset
                let source_dataset = Dataset::open_ex(
                    url,
                    DatasetOptions {
                        open_flags: GdalOpenFlags::GDAL_OF_RASTER | GdalOpenFlags::GDAL_OF_READONLY,
                        ..Default::default()
                    },
                )?;

                // Get the source raster band
                let source_raster_band = source_dataset.rasterband(source_band_num)?;

                // Get the nodata value
                let nodata_value = nodata_bytes_to_f64(band_metadata.nodata_value(), &band_type);

                // Add VRT band
                vrt.add_band(gdal_type, None)?;

                // Get the newly added VRT band
                let vrt_band = vrt.rasterband(i)?;

                // Set nodata value on the VRT band if present
                if let Some(nodata) = nodata_value {
                    vrt_band.set_no_data_value(nodata)?;
                }

                // Add simple source to the VRT band
                vrt_band.add_simple_source(
                    &source_raster_band,
                    (0, 0, width, height), // source window
                    (0, 0, width, height), // destination window
                    None,                  // resampling (default: nearest)
                    nodata_value,          // nodata value
                )?;

                // Keep the source dataset alive
                source_datasets.push(source_dataset);
            }
            StorageType::InDb => {
                return Err(GdalError::BadArgument(format!(
                    "Band {} uses InDb storage which is not supported in VRT datasets. \
                     For mixed storage rasters, either convert in-db bands to out-db references \
                     or use raster_to_mem_dataset() for pure in-db rasters.",
                    i
                )));
            }
        }
    }

    Ok(RasterVrtDataset {
        vrt,
        _source_datasets: source_datasets,
    })
}

/// Creates an appropriate GDAL dataset based on the raster's storage type.
///
/// This is a convenience function that automatically chooses the right conversion
/// method based on whether the raster uses in-db or out-db storage:
/// - Pure in-db rasters: Uses `raster_to_mem_dataset` for zero-copy access
/// - Rasters with any out-db bands: Uses `raster_to_vrt_dataset`
///
/// # Arguments
/// * `raster` - The RasterRef to convert
///
/// # Returns
/// A GDAL Dataset. For in-db rasters, this is a MEM dataset; for out-db rasters,
/// this is a VRT dataset.
///
/// # Note
/// For in-db rasters, the returned Dataset internally references the original
/// RasterRef's memory. Ensure the RasterRef outlives the Dataset.
pub fn raster_to_dataset(raster: &dyn RasterRef) -> Result<Dataset> {
    let bands = raster.bands();

    // Check if any band uses out-db storage
    let has_outdb = (1..=bands.len()).any(|i| {
        bands
            .band(i)
            .map(|b| b.metadata().storage_type() == StorageType::OutDbRef)
            .unwrap_or(false)
    });

    if has_outdb {
        let vrt_dataset = raster_to_vrt_dataset(raster)?;
        // Safety: We're transferring ownership to the caller
        Ok(unsafe { vrt_dataset.into_dataset() })
    } else {
        let mem_dataset = raster_to_mem_dataset(raster)?;
        // Safety: We're transferring ownership to the caller, who must ensure
        // the RasterRef outlives this Dataset
        Ok(unsafe { mem_dataset.into_dataset() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_band_data_type_to_gdal() {
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::UInt8),
            GdalDataType::UInt8
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::UInt16),
            GdalDataType::UInt16
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::Int16),
            GdalDataType::Int16
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::UInt32),
            GdalDataType::UInt32
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::Int32),
            GdalDataType::Int32
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::Float32),
            GdalDataType::Float32
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::Float64),
            GdalDataType::Float64
        );
    }

    #[test]
    fn test_nodata_bytes_to_f64() {
        // UInt8
        assert_eq!(
            nodata_bytes_to_f64(Some(&[255u8]), &BandDataType::UInt8),
            Some(255.0)
        );
        assert_eq!(
            nodata_bytes_to_f64(Some(&[0u8]), &BandDataType::UInt8),
            Some(0.0)
        );

        // Int16
        let val: i16 = -32768;
        assert_eq!(
            nodata_bytes_to_f64(Some(&val.to_le_bytes()), &BandDataType::Int16),
            Some(-32768.0)
        );

        // Float32
        let val: f32 = -9999.0;
        assert_eq!(
            nodata_bytes_to_f64(Some(&val.to_le_bytes()), &BandDataType::Float32),
            Some(-9999.0)
        );

        // Float64
        let val: f64 = std::f64::NAN;
        let result = nodata_bytes_to_f64(Some(&val.to_le_bytes()), &BandDataType::Float64);
        assert!(result.unwrap().is_nan());

        // None input
        assert_eq!(nodata_bytes_to_f64(None, &BandDataType::UInt8), None);

        // Wrong length
        assert_eq!(
            nodata_bytes_to_f64(Some(&[1, 2, 3]), &BandDataType::UInt8),
            None
        );
    }
}
