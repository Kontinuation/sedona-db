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

use criterion::{criterion_group, criterion_main, Criterion};
use sedona_testing::benchmark_util::{benchmark, BenchmarkArgSpec::*, BenchmarkArgs};

fn criterion_benchmark(c: &mut Criterion) {
    let f = sedona_raster_functions::register::default_function_set();

    // RS_GeoReference(raster) - default GDAL format
    benchmark::scalar(c, &f, "rs_georeference", "rs_georeference", Raster(64, 64));

    // RS_GeoReference(raster, format) - ESRI format
    benchmark::scalar(
        c,
        &f,
        "rs_georeference",
        "rs_georeference",
        BenchmarkArgs::ArrayScalar(Raster(64, 64), String("ESRI".to_string())),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
