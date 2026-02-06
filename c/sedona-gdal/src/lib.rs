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

pub mod error;
pub mod gdal;
pub mod gdal_dyn_bindgen;
pub mod register;

mod shim_exports {
    use std::ffi::c_void;
    use std::os::raw::c_int;

    use crate::gdal_dyn_bindgen::{GDALDataType, GDALDatasetH, GSpacing};

    extern "C" {
        fn sedona_gdal_mem_create_internal(
            x_size: c_int,
            y_size: c_int,
            band_count: c_int,
            band_types: *const GDALDataType,
            band_data: *const *const c_void,
            pixel_offsets: *const GSpacing,
            line_offsets: *const GSpacing,
        ) -> GDALDatasetH;
        fn sedona_gdal_dataset_close(dataset: GDALDatasetH);
    }

    #[used]
    static FORCE_SEDONA_GDAL_MEM_CREATE_INTERNAL: unsafe extern "C" fn(
        c_int,
        c_int,
        c_int,
        *const GDALDataType,
        *const *const c_void,
        *const GSpacing,
        *const GSpacing,
    ) -> GDALDatasetH = sedona_gdal_mem_create_internal;

    #[used]
    static FORCE_SEDONA_GDAL_DATASET_CLOSE: unsafe extern "C" fn(GDALDatasetH) =
        sedona_gdal_dataset_close;
}
