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
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_int, c_void};

pub type GSpacing = i64;

#[repr(C)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum GDALDataType {
    GDT_Unknown = 0,
    GDT_Byte = 1,
    GDT_Int8 = 14,
    GDT_UInt16 = 2,
    GDT_Int16 = 3,
    GDT_UInt32 = 4,
    GDT_Int32 = 5,
    GDT_UInt64 = 12,
    GDT_Int64 = 13,
    GDT_Float16 = 15,
    GDT_Float32 = 6,
    GDT_Float64 = 7,
    GDT_CInt16 = 8,
    GDT_CInt32 = 9,
    GDT_CFloat16 = 16,
    GDT_CFloat32 = 10,
    GDT_CFloat64 = 11,
    GDT_TypeCount = 17,
}

pub type GDALDatasetH = *mut c_void;

impl TryFrom<gdal::raster::GdalDataType> for GDALDataType {
    type Error = ();

    fn try_from(value: gdal::raster::GdalDataType) -> Result<Self, Self::Error> {
        let mapped = match value {
            gdal::raster::GdalDataType::Unknown => GDALDataType::GDT_Unknown,
            gdal::raster::GdalDataType::UInt8 => GDALDataType::GDT_Byte,
            gdal::raster::GdalDataType::Int8 => GDALDataType::GDT_Int8,
            gdal::raster::GdalDataType::UInt16 => GDALDataType::GDT_UInt16,
            gdal::raster::GdalDataType::Int16 => GDALDataType::GDT_Int16,
            gdal::raster::GdalDataType::UInt32 => GDALDataType::GDT_UInt32,
            gdal::raster::GdalDataType::Int32 => GDALDataType::GDT_Int32,
            gdal::raster::GdalDataType::UInt64 => GDALDataType::GDT_UInt64,
            gdal::raster::GdalDataType::Int64 => GDALDataType::GDT_Int64,
            gdal::raster::GdalDataType::Float32 => GDALDataType::GDT_Float32,
            gdal::raster::GdalDataType::Float64 => GDALDataType::GDT_Float64,
        };
        Ok(mapped)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SedonaGdalApi {
    pub sedona_gdal_mem_create_internal: Option<
        unsafe extern "C" fn(
            x_size: c_int,
            y_size: c_int,
            band_count: c_int,
            band_types: *const GDALDataType,
            band_data: *const *const c_void,
            pixel_offsets: *const GSpacing,
            line_offsets: *const GSpacing,
        ) -> GDALDatasetH,
    >,
    pub sedona_gdal_dataset_close: Option<unsafe extern "C" fn(dataset: GDALDatasetH)>,
    pub release: Option<unsafe extern "C" fn(arg1: *mut SedonaGdalApi)>,
    pub private_data: *mut c_void,
}

unsafe extern "C" {
    pub fn sedona_gdal_dyn_api_init(
        api: *mut SedonaGdalApi,
        shared_object_path: *const c_char,
        err_msg: *mut c_char,
        len: c_int,
    ) -> c_int;

    pub fn sedona_gdal_dyn_api_init_from_current_process(
        api: *mut SedonaGdalApi,
        err_msg: *mut c_char,
        len: c_int,
    ) -> c_int;
}
