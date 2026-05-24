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

use datafusion::arrow::array::{Array, StringArray};
use sedona::context::SedonaContext;
use sedona_gdal::global::{configure_global_gdal_api, GdalApiBuilder};
use sedona_testing::data::test_raster;

fn configure_test_gdal() {
    let shared_library_path = std::env::var("GDAL_SHARED_LIBRARY")
        .ok()
        .or_else(|| {
            [
                "/lib64/libgdal.so.37.3.11.5",
                "/usr/lib64/libgdal.so.37.3.11.5",
                "/lib64/libgdal.so.37",
                "/usr/lib64/libgdal.so.37",
                "/lib64/libgdal.so",
                "/usr/lib64/libgdal.so",
                "/lib/libgdal.so",
                "/usr/lib/libgdal.so",
                "/usr/local/lib/libgdal.so",
                "/usr/lib/x86_64-linux-gnu/libgdal.so",
            ]
            .into_iter()
            .find(|path| std::path::Path::new(path).exists())
            .map(str::to_string)
        })
        .expect("A usable libgdal shared library should exist for this test");

    configure_global_gdal_api(
        GdalApiBuilder::default().with_shared_library(shared_library_path.into()),
    )
    .expect("GDAL shared library should configure successfully");
}

async fn query_band_paths(sql: &str) -> StringArray {
    let ctx = SedonaContext::new();
    let batches = ctx
        .sql(sql)
        .await
        .expect("SQL planning should succeed")
        .collect()
        .await
        .expect("SQL execution should succeed");

    assert_eq!(batches.len(), 1);
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray result")
        .clone()
}

#[tokio::test]
async fn rs_frompath_sql() {
    configure_test_gdal();
    let path = test_raster("test4.tiff").expect("test4.tiff should exist");

    let array = query_band_paths(&format!(
        "SELECT RS_BandPath(rs_frompath('{path}')) AS path"
    ))
    .await;
    assert_eq!(array.len(), 1);
    assert_eq!(array.value(0), path);
}

#[tokio::test]
async fn rs_frompath_sql_propagates_nulls() {
    configure_test_gdal();
    let path = test_raster("test4.tiff").expect("test4.tiff should exist");

    let array = query_band_paths(&format!(
        "SELECT RS_BandPath(rs_frompath(path)) AS path FROM (VALUES ('{path}'), (NULL)) AS t(path)"
    ))
    .await;
    assert_eq!(array.len(), 2);
    assert_eq!(array.value(0), path);
    assert!(array.is_null(1));
}

#[tokio::test]
async fn rs_frompath_sql_missing_path_errors() {
    configure_test_gdal();
    let ctx = SedonaContext::new();
    let err = ctx
        .sql("SELECT RS_FromPath('/definitely/missing/rs_from_path_test.tif')")
        .await
        .expect("SQL planning should succeed")
        .collect()
        .await
        .expect_err("Missing path should return an error");

    let err_message = err.to_string();
    assert!(err_message.contains(
        "Failed to open raster file '/definitely/missing/rs_from_path_test.tif' (GDAL path '/definitely/missing/rs_from_path_test.tif')"
    ));
}
