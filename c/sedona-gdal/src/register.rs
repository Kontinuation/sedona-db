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

use crate::error::SedonaGdalError;
use crate::gdal::GdalApi;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

static GDAL_API: OnceLock<Arc<GdalApi>> = OnceLock::new();

pub fn configure_global_gdal_api(shared_library: PathBuf) -> Result<(), SedonaGdalError> {
    let api = GdalApi::try_from_shared_library(shared_library)?;
    GDAL_API
        .set(api)
        .map_err(|_| SedonaGdalError::Invalid("GDAL API already configured".to_string()))?;
    Ok(())
}

pub fn configure_global_gdal_api_from_current_process() -> Result<(), SedonaGdalError> {
    let api = GdalApi::try_from_current_process()?;
    GDAL_API
        .set(api)
        .map_err(|_| SedonaGdalError::Invalid("GDAL API already configured".to_string()))?;
    Ok(())
}

pub fn is_gdal_api_configured() -> bool {
    GDAL_API.get().is_some()
}

pub fn with_global_gdal_api<F, R>(func: F) -> Result<R, SedonaGdalError>
where
    F: FnOnce(&Arc<GdalApi>) -> Result<R, SedonaGdalError>,
{
    let api = GDAL_API
        .get()
        .ok_or_else(|| SedonaGdalError::Invalid("GDAL API not configured".to_string()))?;
    func(api)
}
