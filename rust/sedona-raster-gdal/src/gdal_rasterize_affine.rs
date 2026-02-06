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

//! Fast affine-transformer rasterize wrapper.
//!
//! GDALRasterizeGeometries() will internally call GDALCreateGenImgProjTransformer2()
//! if pfnTransformer is NULL, even in the common case where only a GeoTransform-based
//! affine conversion from georeferenced coords to pixel/line is needed.
//!
//! This module supplies a minimal GDALTransformerFunc that applies the dataset
//! GeoTransform (and its inverse), avoiding expensive transformer creation.

use std::ffi::{c_int, c_void};
use std::ptr;

use gdal::errors::{GdalError, Result};
use gdal::vector::Geometry;
use gdal::{Dataset, GeoTransform, GeoTransformEx};
use gdal_sys::CPLErr;

#[repr(C)]
struct AffineTransformArg {
    gt: GeoTransform,
    inv_gt: GeoTransform,
}

unsafe extern "C" fn affine_transformer(
    p_transformer_arg: *mut c_void,
    b_dst_to_src: c_int,
    n_point_count: c_int,
    x: *mut f64,
    y: *mut f64,
    _z: *mut f64,
    pan_success: *mut c_int,
) -> c_int {
    if p_transformer_arg.is_null() || x.is_null() || y.is_null() || pan_success.is_null() {
        return 0;
    }
    if n_point_count < 0 {
        return 0;
    }

    // Treat transformer arg as immutable.
    let arg = &*(p_transformer_arg as *const AffineTransformArg);
    let (t0, t1, t2, t3, t4, t5) = if b_dst_to_src == 0 {
        // Source->destination in GDAL terms for rasterize: world/georef -> pixel/line.
        (
            arg.inv_gt[0],
            arg.inv_gt[1],
            arg.inv_gt[2],
            arg.inv_gt[3],
            arg.inv_gt[4],
            arg.inv_gt[5],
        )
    } else {
        // Destination->source: pixel/line -> world/georef.
        (
            arg.gt[0], arg.gt[1], arg.gt[2], arg.gt[3], arg.gt[4], arg.gt[5],
        )
    };

    let n = n_point_count as usize;
    for i in 0..n {
        // SAFETY: x/y/pan_success are assumed to point to arrays of length n_point_count.
        let xin = unsafe { *x.add(i) };
        let yin = unsafe { *y.add(i) };
        let xout = t0 + xin * t1 + yin * t2;
        let yout = t3 + xin * t4 + yin * t5;
        unsafe {
            *x.add(i) = xout;
            *y.add(i) = yout;
            *pan_success.add(i) = 1;
        }
    }

    1
}

fn last_cpl_err(cpl_err_class: CPLErr::Type, fallback_msg: &str) -> GdalError {
    let last_err_no = unsafe { gdal_sys::CPLGetLastErrorNo() };
    let last_err_msg_ptr = unsafe { gdal_sys::CPLGetLastErrorMsg() };
    let last_err_msg = if last_err_msg_ptr.is_null() {
        None
    } else {
        // SAFETY: CPLGetLastErrorMsg returns a NUL-terminated C string.
        unsafe { std::ffi::CStr::from_ptr(last_err_msg_ptr) }
            .to_str()
            .ok()
            .map(|s| s.to_string())
    };
    unsafe { gdal_sys::CPLErrorReset() };
    GdalError::CplError {
        class: cpl_err_class,
        number: last_err_no,
        msg: last_err_msg.unwrap_or_else(|| fallback_msg.to_string()),
    }
}

/// Rasterize geometries with an affine transformer derived from the destination dataset.
///
/// This mirrors `gdal::raster::rasterize()` but avoids GDAL's slow default transformer creation.
///
/// Assumptions:
/// - Geometry coordinates are already in the destination dataset georeferenced coordinate space.
/// - Only GeoTransform-based affine conversion is supported (no GCP/RPC/geolocs).
pub fn rasterize_affine(
    dataset: &mut Dataset,
    bands: &[usize],
    geometries: &[Geometry],
    burn_values: &[f64],
    all_touched: bool,
) -> Result<()> {
    if bands.is_empty() {
        return Err(GdalError::BadArgument(
            "`bands` must not be empty".to_string(),
        ));
    }
    if burn_values.len() != geometries.len() {
        return Err(GdalError::BadArgument(format!(
            "Burn values length ({}) must match geometries length ({})",
            burn_values.len(),
            geometries.len()
        )));
    }

    let raster_count = dataset.raster_count();
    for band in bands {
        let is_good = *band > 0 && *band <= raster_count;
        if !is_good {
            return Err(GdalError::BadArgument(format!(
                "Band index {} is out of bounds",
                *band
            )));
        }
    }

    let bands_i32: Vec<c_int> = bands.iter().map(|&band| band as c_int).collect();

    let c_options = if all_touched {
        [c"ALL_TOUCHED=TRUE".as_ptr(), ptr::null_mut()]
    } else {
        [c"ALL_TOUCHED=FALSE".as_ptr(), ptr::null_mut()]
    };

    let geometries_c: Vec<_> = geometries
        .iter()
        .map(|geo| unsafe { geo.c_geometry() })
        .collect();
    let burn_values_expanded: Vec<f64> = burn_values
        .iter()
        .flat_map(|burn| std::iter::repeat_n(burn, bands_i32.len()))
        .copied()
        .collect();

    let gt = dataset.geo_transform().map_err(|_e| {
        GdalError::BadArgument(
            "Missing geotransform: only geotransform-based affine rasterize is supported"
                .to_string(),
        )
    })?;
    let inv_gt = gt.invert().map_err(|_e| {
        GdalError::BadArgument(
            "Non-invertible geotransform: only geotransform-based affine rasterize is supported"
                .to_string(),
        )
    })?;
    let mut arg = AffineTransformArg { gt, inv_gt };

    unsafe {
        let error = gdal_sys::GDALRasterizeGeometries(
            dataset.c_dataset(),
            bands_i32.len() as c_int,
            bands_i32.as_ptr(),
            geometries_c.len() as c_int,
            geometries_c.as_ptr(),
            Some(affine_transformer),
            (&mut arg as *mut AffineTransformArg).cast::<c_void>(),
            burn_values_expanded.as_ptr(),
            c_options.as_ptr() as *mut *mut i8,
            None,
            ptr::null_mut(),
        );
        if error != CPLErr::CE_None {
            return Err(last_cpl_err(error, "GDALRasterizeGeometries failed"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gdal_common::mem_driver;
    use gdal::raster::{Buffer, RasterizeOptions};

    fn make_dataset_u8(
        width: usize,
        height: usize,
        gt: GeoTransform,
    ) -> gdal::errors::Result<Dataset> {
        let driver = mem_driver().unwrap();
        let mut ds = driver.create_with_band_type::<u8, _>("", width, height, 1)?;
        ds.set_geo_transform(&gt)?;
        let mut band = ds.rasterband(1)?;
        let mut buf = Buffer::new((width, height), vec![0u8; width * height]);
        band.write((0, 0), (width, height), &mut buf)?;
        Ok(ds)
    }

    fn read_u8(ds: &Dataset, width: usize, height: usize) -> Vec<u8> {
        let band = ds.rasterband(1).unwrap();
        let buf = band
            .read_as::<u8>((0, 0), (width, height), (width, height), None)
            .unwrap();
        buf.data().to_vec()
    }

    fn poly_from_pixel_rect(gt: &GeoTransform, x0: f64, y0: f64, x1: f64, y1: f64) -> Geometry {
        // Build a polygon from pixel/line coords by transforming to georef coords.
        let (wx0, wy0) = gt.apply(x0, y0);
        let (wx1, wy1) = gt.apply(x1, y0);
        let (wx2, wy2) = gt.apply(x1, y1);
        let (wx3, wy3) = gt.apply(x0, y1);
        let wkt =
            format!("POLYGON (({wx0} {wy0}, {wx1} {wy1}, {wx2} {wy2}, {wx3} {wy3}, {wx0} {wy0}))");
        Geometry::from_wkt(&wkt).unwrap()
    }

    fn line_from_pixel_points(gt: &GeoTransform, pts: &[(f64, f64)]) -> Geometry {
        assert!(pts.len() >= 2);
        let mut s = String::from("LINESTRING (");
        for (i, (px, py)) in pts.iter().copied().enumerate() {
            let (wx, wy) = gt.apply(px, py);
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("{wx} {wy}"));
        }
        s.push(')');
        Geometry::from_wkt(&s).unwrap()
    }

    #[test]
    fn test_rasterize_affine_matches_baseline_north_up() {
        let (w, h) = (32usize, 24usize);
        let gt: GeoTransform = [100.0, 2.0, 0.0, 200.0, 0.0, -2.0];

        let geom = poly_from_pixel_rect(&gt, 3.2, 4.7, 20.4, 18.1);
        let opts = RasterizeOptions {
            all_touched: false,
            ..Default::default()
        };

        let mut ds_baseline = make_dataset_u8(w, h, gt).unwrap();
        let mut ds_affine = make_dataset_u8(w, h, gt).unwrap();

        gdal::raster::rasterize(&mut ds_baseline, &[1], &[geom.clone()], &[1.0], Some(opts))
            .unwrap();
        rasterize_affine(&mut ds_affine, &[1], &[geom], &[1.0], false).unwrap();

        assert_eq!(read_u8(&ds_affine, w, h), read_u8(&ds_baseline, w, h));
    }

    #[test]
    fn test_rasterize_affine_matches_baseline_rotated_gt_all_touched() {
        let (w, h) = (40usize, 28usize);
        // Rotated/skewed GeoTransform.
        let gt: GeoTransform = [10.0, 1.2, 0.15, 50.0, -0.1, -1.1];

        let geom = poly_from_pixel_rect(&gt, 5.25, 4.5, 25.75, 20.25);
        let opts = RasterizeOptions {
            all_touched: true,
            ..Default::default()
        };

        let mut ds_baseline = make_dataset_u8(w, h, gt).unwrap();
        let mut ds_affine = make_dataset_u8(w, h, gt).unwrap();

        gdal::raster::rasterize(&mut ds_baseline, &[1], &[geom.clone()], &[1.0], Some(opts))
            .unwrap();
        rasterize_affine(&mut ds_affine, &[1], &[geom], &[1.0], true).unwrap();

        assert_eq!(read_u8(&ds_affine, w, h), read_u8(&ds_baseline, w, h));
    }

    #[test]
    fn test_rasterize_affine_matches_baseline_linestring() {
        let (w, h) = (64usize, 48usize);
        // Rotated/skewed GeoTransform.
        let gt: GeoTransform = [5.0, 1.0, 0.2, 100.0, -0.15, -1.05];

        // A polyline with many vertices, defined in pixel/line space.
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for i in 0..200 {
            let t = i as f64 / 199.0;
            // Avoid coordinates that land extremely close to pixel boundaries; the baseline
            // transformer path can differ by tiny floating point epsilons on some platforms,
            // which may flip a single touched pixel for thin lines.
            let x = 2.625 + t * ((w as f64) - 5.25);
            let y = 5.25 + (t * 6.0).sin() * 8.0 + t * ((h as f64) - 12.25);
            pts.push((x, y));
        }
        let geom = line_from_pixel_points(&gt, &pts);

        let opts = RasterizeOptions {
            all_touched: false,
            ..Default::default()
        };

        let mut ds_baseline = make_dataset_u8(w, h, gt).unwrap();
        let mut ds_affine = make_dataset_u8(w, h, gt).unwrap();

        gdal::raster::rasterize(&mut ds_baseline, &[1], &[geom.clone()], &[1.0], Some(opts))
            .unwrap();
        rasterize_affine(&mut ds_affine, &[1], &[geom], &[1.0], false).unwrap();

        let got = read_u8(&ds_affine, w, h);
        let expected = read_u8(&ds_baseline, w, h);
        if got != expected {
            let mut diffs = Vec::new();
            for (i, (a, b)) in got
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .enumerate()
            {
                if a != b {
                    let x = i % w;
                    let y = i / w;
                    diffs.push((x, y, a, b));
                }
            }
            panic!(
                "raster mismatch: {} differing pixels; first 10: {:?}",
                diffs.len(),
                &diffs[..diffs.len().min(10)]
            );
        }
    }

    #[test]
    fn test_rasterize_affine_fails_on_noninvertible_gt() {
        let (w, h) = (8usize, 8usize);
        let gt: GeoTransform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut ds = make_dataset_u8(w, h, gt).unwrap();
        let geom = Geometry::from_wkt("POINT (0 0)").unwrap();
        let err = rasterize_affine(&mut ds, &[1], &[geom], &[1.0], true).unwrap_err();
        match err {
            GdalError::BadArgument(msg) => {
                assert!(msg.contains("Non-invertible geotransform"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }
}
