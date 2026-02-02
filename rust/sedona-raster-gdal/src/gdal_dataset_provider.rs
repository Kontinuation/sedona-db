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

use std::{cell::RefCell, marker::PhantomData, num::NonZeroUsize, rc::Rc};

use datafusion_common::{arrow_datafusion_err, DataFusionError, Result};

use gdal::GeoTransformEx;

use sedona_raster::traits::RasterRef;
use sedona_schema::raster::StorageType;

use crate::gdal_common::{
    band_data_type_to_gdal, bytes_to_f64, convert_gdal_err, create_outdb_source,
    raster_ref_to_gdal_empty, raster_ref_to_gdal_mem,
};

/// A GDAL dataset constructed from a `RasterRef`.
///
/// This struct is designed to keep any backing GDAL datasets alive for as long as
/// the returned `dataset` might reference them.
///
/// Field semantics by raster storage layout:
///
/// 1) **In-db bands only**
///    - `dataset`: a GDAL **MEM** dataset containing all bands.
///    - `gdal_mem_source`: `None` (the MEM dataset is already stored in `dataset`).
///    - `_gdal_outdb_sources`: empty.
///
/// 2) **Out-db bands only**
///    - `dataset`: a GDAL **VRT** dataset sized like the target raster, with each VRT band
///      sourcing from an external dataset band.
///    - `gdal_mem_source`: `None`.
///    - `_gdal_outdb_sources`: contains the opened external GDAL datasets (kept alive via `Rc`).
///      (There may be duplicates if multiple bands reference the same URL; that is fine.)
///
/// 3) **Mixed in-db + out-db bands**
///    - `dataset`: a GDAL **VRT** dataset with band order matching the target raster.
///      In-db bands source from a MEM dataset; out-db bands source from external datasets.
///    - `gdal_mem_source`: `Some(MEM dataset)` containing only the in-db bands, in the same order
///      as they appear in the target raster.
///    - `_gdal_outdb_sources`: contains the opened external GDAL datasets (kept alive via `Rc`).
pub(crate) struct RasterDataset<'a> {
    /// The dataset to use for further GDAL operations.
    dataset: gdal::Dataset,
    /// A MEM dataset holding in-db band data when `dataset` is a VRT that references it.
    _gdal_mem_source: Option<gdal::Dataset>,
    /// External datasets referenced by the VRT; kept alive for the lifetime of this struct.
    _gdal_outdb_sources: Vec<Rc<gdal::Dataset>>,
    _phantom: PhantomData<&'a ()>,
}

impl<'a> RasterDataset<'a> {
    /// Return a reference to the underlying GDAL dataset.
    pub(crate) fn as_dataset(&self) -> &gdal::Dataset {
        &self.dataset
    }
}

thread_local! {
    /// Thread-local lazily-initialized `GDALDatasetProvider`.
    static TL_GDAL_PROVIDER: RefCell<Option<Rc<GDALDatasetProvider>>> = const { RefCell::new(None) };
}

/// Get or create the thread-local `GDALDatasetProvider`.
pub(crate) fn thread_local_provider() -> Result<Rc<GDALDatasetProvider>> {
    TL_GDAL_PROVIDER.with(|cell| {
        let mut opt = cell.borrow_mut();
        if let Some(rc) = opt.as_ref() {
            Ok(Rc::clone(rc))
        } else {
            // Cache size chosen modestly; can be tuned per workload.
            let provider = Rc::new(GDALDatasetProvider::try_new(32)?);
            *opt = Some(Rc::clone(&provider));
            Ok(provider)
        }
    })
}

#[derive(Hash, Eq, PartialEq)]
struct OutDbSourceKey {
    path: String,
    open_options: Option<Vec<String>>,
}

impl OutDbSourceKey {
    pub fn new(path: &str, options: Option<&[&str]>) -> Self {
        let open_options = options
            .filter(|opts| !opts.is_empty())
            .map(|opts| opts.iter().map(|s| (*s).to_string()).collect());

        Self {
            path: path.to_string(),
            open_options,
        }
    }
}

pub(crate) struct GDALDatasetProvider {
    cached_sources: RefCell<lru::LruCache<OutDbSourceKey, Rc<gdal::Dataset>>>,
}

impl GDALDatasetProvider {
    pub fn try_new(cache_capacity: usize) -> Result<Self> {
        let Some(cap) = NonZeroUsize::new(cache_capacity) else {
            return Err(DataFusionError::Configuration(
                "Raster source cache size should be greater than 0".to_string(),
            ));
        };
        let cache = lru::LruCache::new(cap);
        Ok(Self {
            cached_sources: RefCell::new(cache),
        })
    }

    pub fn raster_ref_to_gdal<'a, R: RasterRef>(&self, raster: &'a R) -> Result<RasterDataset<'a>> {
        let metadata = raster.metadata();
        let bands = raster.bands();
        let num_bands = bands.len();

        if num_bands == 0 {
            let dataset = raster_ref_to_gdal_empty(raster)?;
            return Ok(RasterDataset {
                dataset,
                _gdal_mem_source: None,
                _gdal_outdb_sources: Vec::new(),
                _phantom: PhantomData,
            });
        }

        let mut indb_band_indices = Vec::with_capacity(num_bands);
        let mut has_outdb = false;
        for i in 1..=num_bands {
            let band = bands.band(i).map_err(|e| arrow_datafusion_err!(e))?;
            match band.metadata().storage_type() {
                StorageType::InDb => indb_band_indices.push(i),
                StorageType::OutDbRef => has_outdb = true,
            }
        }

        let mut gdal_mem_source = if !indb_band_indices.is_empty() {
            Some(unsafe { raster_ref_to_gdal_mem(raster, &indb_band_indices)? })
        } else {
            None
        };

        // Pure in-db: the MEM dataset is the final dataset.
        if !has_outdb {
            let dataset = gdal_mem_source.take().expect("in-db dataset should exist");
            return Ok(RasterDataset {
                dataset,
                _gdal_mem_source: None,
                _gdal_outdb_sources: Vec::new(),
                _phantom: PhantomData,
            });
        }

        // Mixed or pure out-db: build a VRT dataset referencing sources.
        let width = metadata.width() as i32;
        let height = metadata.height() as i32;
        let mut vrt =
            gdal::vrt::VrtDataset::create(metadata.width() as usize, metadata.height() as usize)
                .map_err(convert_gdal_err)?;

        let geotransform = [
            metadata.upper_left_x(),
            metadata.scale_x(),
            metadata.skew_x(),
            metadata.upper_left_y(),
            metadata.skew_y(),
            metadata.scale_y(),
        ];
        vrt.set_geo_transform(&geotransform)
            .map_err(convert_gdal_err)?;
        if let Some(crs) = raster.crs() {
            vrt.set_projection(crs).map_err(convert_gdal_err)?;
        }

        let mut outdb_sources: Vec<Rc<gdal::Dataset>> = Vec::new();
        let mut mem_band_index = 1usize;

        for i in 1..=num_bands {
            let band = bands.band(i).map_err(|e| arrow_datafusion_err!(e))?;
            let band_metadata = band.metadata();
            let band_type = band_metadata.data_type();
            let gdal_type = band_data_type_to_gdal(&band_type);

            let nodata_value = band_metadata
                .nodata_value()
                .map(|bytes| bytes_to_f64(bytes, &band_type))
                .transpose()?;

            vrt.add_band(gdal_type, None).map_err(convert_gdal_err)?;
            let vrt_band = vrt.rasterband(i).map_err(convert_gdal_err)?;

            if let Some(nodata) = nodata_value {
                vrt_band
                    .set_no_data_value(nodata)
                    .map_err(convert_gdal_err)?;
            }

            match band_metadata.storage_type() {
                StorageType::OutDbRef => {
                    let url = band_metadata.outdb_url().ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Band {} is out-db but missing outdb_url",
                            i
                        ))
                    })?;
                    let source_band_num = band_metadata.outdb_band_id().ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Band {} is out-db but missing band_id",
                            i
                        ))
                    })?;

                    let source_dataset = self.get_or_create_outdb_source(url, None)?;

                    // If GDALGetGeoTransform(hdsSrc, ogt) fails, we falls back to (0, 1, 0, 0, 0, -1),
                    // which is the identity transform.
                    let src_geo_transform = source_dataset
                        .geo_transform()
                        .unwrap_or([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]);
                    let (src_w, src_h) = source_dataset.raster_size();

                    // Compute source and destination windows for the VRT simple source. The VRT usually only
                    // clip a small portion of the source dataset.
                    let Some((src_window, dst_window)) = compute_vrt_simple_source_windows(
                        &geotransform,
                        (width, height),
                        &src_geo_transform,
                        (src_w as i32, src_h as i32),
                    )?
                    else {
                        // No spatial overlap between the target raster and the source dataset.
                        // Leave the VRT band as nodata.
                        continue;
                    };

                    let source_band = source_dataset
                        .rasterband(source_band_num as usize)
                        .map_err(convert_gdal_err)?;

                    vrt_band
                        .add_simple_source(&source_band, src_window, dst_window, None, nodata_value)
                        .map_err(convert_gdal_err)?;

                    outdb_sources.push(source_dataset);
                }
                StorageType::InDb => {
                    let mem_dataset = gdal_mem_source
                        .as_ref()
                        .expect("in-db dataset should exist");
                    let source_band = mem_dataset
                        .rasterband(mem_band_index)
                        .map_err(convert_gdal_err)?;
                    mem_band_index += 1;

                    vrt_band
                        .add_simple_source(
                            &source_band,
                            (0, 0, width, height),
                            (0, 0, width, height),
                            None,
                            nodata_value,
                        )
                        .map_err(convert_gdal_err)?;
                }
            }
        }

        Ok(RasterDataset {
            dataset: vrt.as_dataset(),
            _gdal_mem_source: gdal_mem_source,
            _gdal_outdb_sources: outdb_sources,
            _phantom: PhantomData,
        })
    }

    fn get_or_create_outdb_source(
        &self,
        path: &str,
        options: Option<&[&str]>,
    ) -> Result<Rc<gdal::Dataset>> {
        let cache_key = OutDbSourceKey::new(path, options);
        let mut cache = self.cached_sources.borrow_mut();
        if let Some(cached_source) = cache.get(&cache_key) {
            Ok(Rc::clone(cached_source))
        } else {
            let source_dataset = create_outdb_source(path, options)?;
            let rc_dataset = Rc::new(source_dataset);
            cache.put(cache_key, Rc::clone(&rc_dataset));
            Ok(rc_dataset)
        }
    }
}

type PixelWindow = (i32, i32, i32, i32);
type PixelWindowOverlap = Option<(PixelWindow, PixelWindow)>;

fn compute_vrt_simple_source_windows(
    dst_gt: &gdal::GeoTransform,
    dst_size: (i32, i32),
    src_gt: &gdal::GeoTransform,
    src_size: (i32, i32),
) -> Result<PixelWindowOverlap> {
    let (dst_w, dst_h) = dst_size;
    let (src_w, src_h) = src_size;
    if dst_w <= 0 || dst_h <= 0 || src_w <= 0 || src_h <= 0 {
        return Ok(None);
    }

    // Alignment check (similar intent to rt_raster_same_alignment()).
    // Require equal pixel size + rotation terms.
    let eps = f32::EPSILON as f64;
    if (dst_gt[1] - src_gt[1]).abs() > eps
        || (dst_gt[2] - src_gt[2]).abs() > eps
        || (dst_gt[4] - src_gt[4]).abs() > eps
        || (dst_gt[5] - src_gt[5]).abs() > eps
    {
        return Err(DataFusionError::Execution(format!(
            "Out-db raster is not aligned with target raster (geotransform mismatch): dst={:?} src={:?}",
            dst_gt, src_gt
        )));
    }

    // Compute the pixel/line offset of the destination upper-left in the source grid.
    let inv_src = src_gt.invert().map_err(|e| {
        DataFusionError::Execution(format!("Failed to invert source geotransform: {e}"))
    })?;
    let (off_x_f, off_y_f) = inv_src.apply(dst_gt[0], dst_gt[3]);

    let off_x_r: f64 = off_x_f.round();
    let off_y_r: f64 = off_y_f.round();
    if (off_x_f - off_x_r).abs() > eps || (off_y_f - off_y_r).abs() > eps {
        return Err(DataFusionError::Execution(format!(
            "Out-db raster is not aligned with target raster (non-integer pixel offset): off=({off_x_f},{off_y_f})"
        )));
    }

    if off_x_r < (i32::MIN as f64)
        || off_x_r > (i32::MAX as f64)
        || off_y_r < (i32::MIN as f64)
        || off_y_r > (i32::MAX as f64)
    {
        return Err(DataFusionError::Execution(
            "Out-db raster alignment offset is out of supported range".to_string(),
        ));
    }

    let off_x = off_x_r as i32;
    let off_y = off_y_r as i32;

    // Compute overlapped windows (clipped) while preserving the aligned-grid assumption.
    let dst_xoff = 0.max(-off_x);
    let dst_yoff = 0.max(-off_y);
    let src_xoff = 0.max(off_x);
    let src_yoff = 0.max(off_y);

    let xsize = (dst_w - dst_xoff).min(src_w - src_xoff);
    let ysize = (dst_h - dst_yoff).min(src_h - src_yoff);
    if xsize <= 0 || ysize <= 0 {
        return Ok(None);
    }

    Ok(Some((
        (src_xoff, src_yoff, xsize, ysize),
        (dst_xoff, dst_yoff, xsize, ysize),
    )))
}
