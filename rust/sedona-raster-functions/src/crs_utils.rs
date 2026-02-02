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

use datafusion_common::{DataFusionError, Result};
use sedona_geometry::transform::{transform, CrsEngine};
use sedona_schema::crs::deserialize_crs;
use wkb::reader::read_wkb;

const WGS84_CRS: &str = "EPSG:4326";

pub fn normalize_crs_string(crs: Option<&str>) -> Result<String> {
    let crs_str = crs.unwrap_or(WGS84_CRS);
    if let Ok(Some(crs_ref)) = deserialize_crs(crs_str) {
        return Ok(crs_ref.to_crs_string());
    }
    Ok(crs_str.to_string())
}

pub fn crs_equivalent(a: Option<&str>, b: Option<&str>) -> Result<bool> {
    let a_crs = normalize_crs_string(a)?;
    let b_crs = normalize_crs_string(b)?;
    if a_crs == b_crs {
        return Ok(true);
    }

    let wgs84_codes = ["EPSG:4326", "OGC:CRS84", "CRS84", "WGS84"];
    let a_is_wgs84 = wgs84_codes.iter().any(|c| a_crs.eq_ignore_ascii_case(c));
    let b_is_wgs84 = wgs84_codes.iter().any(|c| b_crs.eq_ignore_ascii_case(c));
    Ok(a_is_wgs84 && b_is_wgs84)
}

pub fn transform_wkb_to_crs(wkb: &[u8], from: Option<&str>, to: Option<&str>) -> Result<Vec<u8>> {
    if crs_equivalent(from, to)? {
        return Ok(wkb.to_vec());
    }

    let from_str = normalize_crs_string(from)?;
    let to_str = normalize_crs_string(to)?;
    let mut out = Vec::with_capacity(wkb.len());
    sedona_proj::st_transform::with_global_proj_engine(|engine| {
        let crs_transform = engine
            .get_transform_crs_to_crs(&from_str, &to_str, None, "")
            .map_err(|e| DataFusionError::Execution(format!("CRS transform error: {e}")))?;
        let geom = read_wkb(wkb).map_err(|e| DataFusionError::External(Box::new(e)))?;
        transform(geom, crs_transform.as_ref(), &mut out)
            .map_err(|e| DataFusionError::Execution(format!("Transform error: {e}")))?;
        Ok(())
    })?;

    Ok(out)
}
