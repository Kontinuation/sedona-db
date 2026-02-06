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

use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion};

use gdal::raster::RasterizeOptions;
use gdal::vector::Geometry;
use gdal::{DriverManager, GeoTransform, GeoTransformEx};

fn ensure_mem_driver() {
    let _ = DriverManager::get_driver_by_name("MEM");
}

fn bench_threads() -> usize {
    if let Ok(s) = std::env::var("SEDONA_BENCH_THREADS") {
        if let Ok(n) = s.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

fn bench_parallel_repeats() -> u64 {
    if let Ok(s) = std::env::var("SEDONA_BENCH_PAR_REPEATS") {
        if let Ok(n) = s.parse::<u64>() {
            if n > 0 {
                return n;
            }
        }
    }
    50
}

fn div_duration(d: Duration, denom: u64) -> Duration {
    if denom == 0 {
        return d;
    }
    Duration::from_secs_f64(d.as_secs_f64() / denom as f64)
}

fn poly_from_pixel_rect(gt: &GeoTransform, x0: f64, y0: f64, x1: f64, y1: f64) -> Geometry {
    let (wx0, wy0) = gt.apply(x0, y0);
    let (wx1, wy1) = gt.apply(x1, y0);
    let (wx2, wy2) = gt.apply(x1, y1);
    let (wx3, wy3) = gt.apply(x0, y1);
    let wkt =
        format!("POLYGON (({wx0} {wy0}, {wx1} {wy1}, {wx2} {wy2}, {wx3} {wy3}, {wx0} {wy0}))");
    Geometry::from_wkt(&wkt).unwrap()
}

fn setup_thread_local_config() {
    // Set frequently requested GDAL config options as thread-local options to eliminate the
    // need for acquiring configs from global config or environment variable, which is very
    // likely to result in heavy contention in multi-threaded environments.
    let thread_local_options = [
        ("CPL_DEBUG", "OFF"),
        ("OSR_DEFAULT_AXIS_MAPPING_STRATEGY", "AUTHORITY_COMPLIANT"),
        ("GDAL_VALIDATE_CREATION_OPTIONS", "YES"),
        ("CHECK_WITH_INVERT_PROJ", "NO"),
        ("GDAL_FORCE_CACHING", "NO"),
        ("GDAL_ENABLE_READ_WRITE_MUTEX", "YES"),
    ];

    for (key, value) in thread_local_options {
        gdal::config::set_thread_local_config_option(key, value)
            .expect(format!("Failed to setup config for {}", key).as_str());
    }
}

fn bench_rasterize_affine(c: &mut Criterion) {
    ensure_mem_driver();

    let driver = DriverManager::get_driver_by_name("MEM").unwrap();
    let (w, h) = (2usize, 2usize);
    // Rotated/skewed GeoTransform.
    let gt: GeoTransform = [10.0, 1.0, 0.0, 50.0, 0.0, -1.0];
    // Small polygon to keep per-call raster work low (so transformer/locks show up more).
    let geom = poly_from_pixel_rect(&gt, 20.25, 30.5, 40.75, 50.25);

    let opts = RasterizeOptions {
        all_touched: true,
        ..Default::default()
    };

    // Do enough calls to make transformer creation cost visible.
    let repeats: usize = 200;

    c.bench_function("rasterize/baseline_null_transformer", |b| {
        let mut ds = driver.create_with_band_type::<u8, _>("", w, h, 1).unwrap();
        ds.set_geo_transform(&gt).unwrap();

        let geoms = [geom.clone()];
        let burns = [1.0f64];

        b.iter(|| {
            for _ in 0..repeats {
                gdal::raster::rasterize(
                    &mut ds,
                    &[1],
                    std::hint::black_box(&geoms),
                    std::hint::black_box(&burns),
                    Some(opts),
                )
                .unwrap();
            }
        })
    });

    c.bench_function("rasterize/affine_transformer", |b| {
        let mut ds = driver.create_with_band_type::<u8, _>("", w, h, 1).unwrap();
        ds.set_geo_transform(&gt).unwrap();

        let geoms = [geom.clone()];
        let burns = [1.0f64];

        b.iter(|| {
            for _ in 0..repeats {
                sedona_raster_gdal::rasterize_affine(
                    &mut ds,
                    &[1],
                    std::hint::black_box(&geoms),
                    std::hint::black_box(&burns),
                    true,
                )
                .unwrap();
            }
        })
    });

    // Parallel: saturate all cores and measure wall-clock throughput.
    // One dataset per thread to focus on GDAL internal/global lock contention.
    let threads = bench_threads();
    let par_repeats = bench_parallel_repeats();
    let wkt = geom.wkt().unwrap();

    let name = format!("rasterize/baseline_null_transformer/parallel_t{threads}_r{par_repeats}");
    c.bench_function(&name, |b| {
        b.iter_custom(|iters| {
            let ready = Arc::new(Barrier::new(threads + 1));
            let start = Arc::new(Barrier::new(threads + 1));

            let mut handles = Vec::with_capacity(threads);
            for _ in 0..threads {
                let ready = ready.clone();
                let start = start.clone();
                let wkt = wkt.clone();
                handles.push(std::thread::spawn(move || {
                    setup_thread_local_config();

                    // Setup (excluded from timing).
                    let driver = DriverManager::get_driver_by_name("MEM").unwrap();
                    let mut ds = driver.create_with_band_type::<u8, _>("", w, h, 1).unwrap();
                    ds.set_geo_transform(&gt).unwrap();

                    let geom = Geometry::from_wkt(&wkt).unwrap();
                    let geoms = [geom];
                    let burns = [1.0f64];
                    let opts = RasterizeOptions {
                        all_touched: true,
                        ..Default::default()
                    };

                    ready.wait();
                    start.wait();

                    for _ in 0..iters {
                        for _ in 0..par_repeats {
                            gdal::raster::rasterize(
                                &mut ds,
                                &[1],
                                std::hint::black_box(&geoms),
                                std::hint::black_box(&burns),
                                Some(opts),
                            )
                            .unwrap();
                        }
                    }
                }));
            }

            ready.wait();
            let t0 = Instant::now();
            start.wait();
            for h in handles {
                h.join().unwrap();
            }

            // Return total duration for `iters` iterations, normalized to a per-call duration.
            // Criterion will divide by `iters` again when reporting time/iter.
            div_duration(t0.elapsed(), threads as u64 * par_repeats)
        })
    });

    let name = format!("rasterize/affine_transformer/parallel_t{threads}_r{par_repeats}");
    c.bench_function(&name, |b| {
        b.iter_custom(|iters| {
            let ready = Arc::new(Barrier::new(threads + 1));
            let start = Arc::new(Barrier::new(threads + 1));

            let mut handles = Vec::with_capacity(threads);
            for _ in 0..threads {
                let ready = ready.clone();
                let start = start.clone();
                let wkt = wkt.clone();
                handles.push(std::thread::spawn(move || {
                    setup_thread_local_config();

                    // Setup (excluded from timing).
                    let driver = DriverManager::get_driver_by_name("MEM").unwrap();
                    let mut ds = driver.create_with_band_type::<u8, _>("", w, h, 1).unwrap();
                    ds.set_geo_transform(&gt).unwrap();

                    let geom = Geometry::from_wkt(&wkt).unwrap();
                    let geoms = [geom];
                    let burns = [1.0f64];

                    ready.wait();
                    start.wait();

                    for _ in 0..iters {
                        for _ in 0..par_repeats {
                            sedona_raster_gdal::rasterize_affine(
                                &mut ds,
                                &[1],
                                std::hint::black_box(&geoms),
                                std::hint::black_box(&burns),
                                true,
                            )
                            .unwrap();
                        }
                    }
                }));
            }

            ready.wait();
            let t0 = Instant::now();
            start.wait();
            for h in handles {
                h.join().unwrap();
            }

            div_duration(t0.elapsed(), threads as u64 * par_repeats)
        })
    });
}

criterion_group!(benches, bench_rasterize_affine);
criterion_main!(benches);
