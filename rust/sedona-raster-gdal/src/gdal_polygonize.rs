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

use std::ffi::CStr;
use std::ptr::null_mut;

use datafusion_common::{DataFusionError, Result};
use gdal::raster::RasterBand;
use gdal::vector::LayerAccess;

/// Safe wrapper around GDAL's `GDALPolygonize`.
///
/// This keeps the raw-handle / unsafe interaction with GDAL confined to a single place.
pub fn polygonize<L: LayerAccess>(
    band: &RasterBand<'_>,
    layer: &L,
    field_index: i32,
) -> Result<()> {
    // SAFETY: We only pass handles obtained from safe `gdal` wrapper objects.
    // GDALPolygonize does not take ownership of the band or layer handles.
    let rv = unsafe {
        gdal_sys::GDALPolygonize(
            band.c_rasterband(),
            null_mut(),
            layer.c_layer(),
            field_index,
            null_mut(),
            None,
            null_mut(),
        )
    };

    if rv == gdal_sys::CPLErr::CE_None {
        return Ok(());
    }

    // Best-effort error message extraction.
    let err_no = unsafe { gdal_sys::CPLGetLastErrorNo() };
    let err_msg = unsafe { gdal_sys::CPLGetLastErrorMsg() };
    let err_msg = if err_msg.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(err_msg) }
            .to_string_lossy()
            .to_string()
    };
    unsafe { gdal_sys::CPLErrorReset() };

    Err(DataFusionError::Execution(format!(
        "GDALPolygonize failed (CPLErr={:?}, errno={}, msg={})",
        rv, err_no, err_msg
    )))
}
