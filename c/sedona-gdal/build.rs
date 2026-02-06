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

fn main() {
    println!("cargo:rerun-if-changed=src/gdal_cpp_shim.cpp");
    println!("cargo:rerun-if-changed=src/gdal_cpp_shim.h");
    println!("cargo:rerun-if-changed=src/gdal_dyn.c");
    println!("cargo:rerun-if-changed=src/gdal_dyn.h");

    let gdal = pkg_config::Config::new()
        .probe("gdal")
        .expect("Failed to find gdal via pkg-config. Set PKG_CONFIG_PATH or GDAL_* env vars.");

    let include_paths = gdal.include_paths;

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/gdal_cpp_shim.cpp")
        .define("SEDONA_GDAL_BUILD", None)
        .include("src")
        .includes(&include_paths)
        .compile("sedona_gdal_cpp_shim");

    cc::Build::new()
        .file("src/gdal_dyn.c")
        .include("src")
        .includes(&include_paths)
        .compile("sedona_gdal_dyn");

    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target == "macos" {
        println!("cargo:rustc-link-lib=c++");
    } else if target == "windows" {
        println!("cargo:rustc-link-lib=msvcrt");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
