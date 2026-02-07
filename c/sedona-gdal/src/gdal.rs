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
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;

use crate::error::SedonaGdalError;
use crate::gdal_dyn_bindgen::{GDALDataType, GDALDatasetH, GSpacing, SedonaGdalApi};

macro_rules! call_gdal_api {
    ($api:expr, $func:ident $(, $arg:expr)*) => {
        if let Some(func) = $api.inner.$func {
            func($($arg),*)
        } else {
            panic!("{} function not available", stringify!($func))
        }
    };
}

#[derive(Debug)]
pub struct GdalApi {
    inner: SedonaGdalApi,
    name: String,
}

unsafe impl Send for GdalApi {}
unsafe impl Sync for GdalApi {}

impl Drop for GdalApi {
    fn drop(&mut self) {
        if let Some(releaser) = self.inner.release {
            unsafe { releaser(&mut self.inner) }
        }
    }
}

impl GdalApi {
    pub fn try_from_shared_library(shared_library: PathBuf) -> Result<Arc<Self>, SedonaGdalError> {
        let mut inner = SedonaGdalApi {
            sedona_gdal_mem_create_internal: None,
            release: None,
            private_data: ptr::null_mut(),
            shim: ptr::null_mut(),
        };
        let mut err_message = (0..1024).map(|_| 0).collect::<Vec<u8>>();
        let shared_library_c = CString::new(shared_library.to_string_lossy().to_string())
            .map_err(|_| SedonaGdalError::Invalid("embedded nul in rust string".to_string()))?;

        let err = unsafe {
            crate::gdal_dyn_bindgen::sedona_gdal_dyn_api_init(
                &mut inner as _,
                shared_library_c.as_ptr(),
                err_message.as_mut_ptr() as _,
                err_message.len().try_into().unwrap(),
            )
        };

        let c_err_message = CStr::from_bytes_until_nul(&err_message)
            .map_err(|_| SedonaGdalError::Invalid("embedded nul in C string".to_string()))?;
        if err != 0 {
            return Err(SedonaGdalError::LibraryError(
                c_err_message.to_string_lossy().to_string(),
            ));
        }

        Ok(Arc::new(Self {
            inner,
            name: shared_library.to_string_lossy().to_string(),
        }))
    }

    pub fn try_from_current_process() -> Result<Arc<Self>, SedonaGdalError> {
        let mut inner = SedonaGdalApi {
            sedona_gdal_mem_create_internal: None,
            release: None,
            private_data: ptr::null_mut(),
            shim: ptr::null_mut(),
        };
        let mut err_message = (0..1024).map(|_| 0).collect::<Vec<u8>>();

        let err = unsafe {
            crate::gdal_dyn_bindgen::sedona_gdal_dyn_api_init_from_current_process(
                &mut inner as _,
                err_message.as_mut_ptr() as _,
                err_message.len().try_into().unwrap(),
            )
        };

        let c_err_message = CStr::from_bytes_until_nul(&err_message)
            .map_err(|_| SedonaGdalError::Invalid("embedded nul in C string".to_string()))?;
        if err != 0 {
            return Err(SedonaGdalError::LibraryError(
                c_err_message.to_string_lossy().to_string(),
            ));
        }

        Ok(Arc::new(Self {
            inner,
            name: "current_process".to_string(),
        }))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub unsafe fn mem_create_internal(
        &self,
        x_size: i32,
        y_size: i32,
        band_count: i32,
        band_types: *const GDALDataType,
        band_data: *const *const std::ffi::c_void,
        pixel_offsets: *const GSpacing,
        line_offsets: *const GSpacing,
    ) -> GDALDatasetH {
        call_gdal_api!(
            self,
            sedona_gdal_mem_create_internal,
            x_size,
            y_size,
            band_count,
            band_types,
            band_data,
            pixel_offsets,
            line_offsets
        )
    }
}
