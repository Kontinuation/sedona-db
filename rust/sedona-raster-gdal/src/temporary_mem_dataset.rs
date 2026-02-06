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
use gdal::raster::GdalType;
use gdal::{Dataset, DatasetOptions, GdalOpenFlags};

use crate::gdal_common::convert_gdal_err;

const GDAL_MEM_ENABLE_OPEN_KEY: &str = "GDAL_MEM_ENABLE_OPEN";
const GDAL_MEM_ENABLE_OPEN_VALUE: &str = "YES";
const GDAL_MEM_ENABLE_OPEN_SENTINEL: &str = "__SEDONA_MEM_OPEN_UNSET__";

struct ThreadLocalConfigGuard {
    key: &'static str,
    previous: Option<String>,
}

impl ThreadLocalConfigGuard {
    fn set(key: &'static str, value: &str) -> Result<Self> {
        let existing =
            gdal::config::get_thread_local_config_option(key, GDAL_MEM_ENABLE_OPEN_SENTINEL)
                .map_err(convert_gdal_err)?;
        let previous = if existing == GDAL_MEM_ENABLE_OPEN_SENTINEL {
            None
        } else {
            Some(existing)
        };
        gdal::config::set_thread_local_config_option(key, value).map_err(convert_gdal_err)?;
        Ok(Self { key, previous })
    }
}

impl Drop for ThreadLocalConfigGuard {
    fn drop(&mut self) {
        let result = if let Some(previous) = self.previous.as_ref() {
            gdal::config::set_thread_local_config_option(self.key, previous)
        } else {
            gdal::config::clear_thread_local_config_option(self.key)
        };
        let _ = result;
    }
}

/// Temporary MEM dataset backed by an application-owned buffer.
///
/// Data is stored in band-sequential (BSQ) order. Each band is stored contiguously:
/// band 1 plane, then band 2 plane, etc.
pub(crate) struct TemporaryMemDataset<T> {
    dataset: Dataset,
    buffer: Vec<T>,
    width: usize,
    height: usize,
    bands: usize,
}

impl<T> TemporaryMemDataset<T>
where
    T: GdalType + Copy + Default,
{
    pub fn new_zeroed(width: usize, height: usize, bands: usize) -> Result<Self> {
        Self::new_filled(width, height, bands, T::default())
    }
}

impl<T> TemporaryMemDataset<T>
where
    T: GdalType + Copy,
{
    pub fn new_filled(width: usize, height: usize, bands: usize, value: T) -> Result<Self> {
        let pixel_count = checked_pixel_count(width, height, bands)?;
        let buffer = vec![value; pixel_count];
        Self::from_buffer(width, height, bands, buffer)
    }

    pub fn from_buffer(width: usize, height: usize, bands: usize, buffer: Vec<T>) -> Result<Self> {
        if width == 0 || height == 0 || bands == 0 {
            return Err(DataFusionError::Execution(
                "Temporary MEM dataset dimensions must be non-zero".to_string(),
            ));
        }
        let pixel_count = checked_pixel_count(width, height, bands)?;
        if buffer.len() != pixel_count {
            return Err(DataFusionError::Execution(format!(
                "Temporary MEM dataset buffer size mismatch: expected {}, got {}",
                pixel_count,
                buffer.len()
            )));
        }

        let mut buffer = buffer;
        let dataset = open_mem_dataset(&mut buffer, width, height, bands)?;

        Ok(Self {
            dataset,
            buffer,
            width,
            height,
            bands,
        })
    }

    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    pub fn dataset_mut(&mut self) -> &mut Dataset {
        &mut self.dataset
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn bands(&self) -> usize {
        self.bands
    }

    pub fn clear(&mut self, value: T) {
        self.buffer.fill(value);
    }

    pub fn band_slice(&self, band_index: usize) -> Result<&[T]> {
        let band = checked_band_index(band_index, self.bands)?;
        let band_size = self.width * self.height;
        let start = band * band_size;
        let end = start + band_size;
        Ok(&self.buffer[start..end])
    }

    pub fn band_slice_mut(&mut self, band_index: usize) -> Result<&mut [T]> {
        let band = checked_band_index(band_index, self.bands)?;
        let band_size = self.width * self.height;
        let start = band * band_size;
        let end = start + band_size;
        Ok(&mut self.buffer[start..end])
    }

    /// Read a window directly from the internal buffer (band-sequential layout).
    pub fn read_as(
        &self,
        band_index: usize,
        offset: (usize, usize),
        size: (usize, usize),
    ) -> Result<Vec<T>> {
        let (xoff, yoff) = offset;
        let (win_w, win_h) = size;
        validate_window(self.width, self.height, xoff, yoff, win_w, win_h)?;

        let band = self.band_slice(band_index)?;
        let mut out = Vec::with_capacity(win_w * win_h);
        for row in 0..win_h {
            let row_start = (yoff + row) * self.width + xoff;
            let row_end = row_start + win_w;
            out.extend_from_slice(&band[row_start..row_end]);
        }
        Ok(out)
    }

    pub fn read_as_into(
        &self,
        band_index: usize,
        offset: (usize, usize),
        size: (usize, usize),
        out: &mut [T],
    ) -> Result<()> {
        let (xoff, yoff) = offset;
        let (win_w, win_h) = size;
        validate_window(self.width, self.height, xoff, yoff, win_w, win_h)?;

        let needed = win_w * win_h;
        if out.len() < needed {
            return Err(DataFusionError::Execution(format!(
                "Output buffer too small: expected {}, got {}",
                needed,
                out.len()
            )));
        }

        let band = self.band_slice(band_index)?;
        for row in 0..win_h {
            let row_start = (yoff + row) * self.width + xoff;
            let row_end = row_start + win_w;
            let out_start = row * win_w;
            let out_end = out_start + win_w;
            out[out_start..out_end].copy_from_slice(&band[row_start..row_end]);
        }
        Ok(())
    }
}

fn open_mem_dataset<T: GdalType + Copy>(
    buffer: &mut [T],
    width: usize,
    height: usize,
    bands: usize,
) -> Result<Dataset> {
    let pixel_size = std::mem::size_of::<T>();
    let pixels_per_band = checked_pixels_per_band(width, height)?;
    let line_offset = width
        .checked_mul(pixel_size)
        .ok_or_else(|| DataFusionError::Execution("Line offset overflow".to_string()))?;
    let band_offset = pixels_per_band
        .checked_mul(pixel_size)
        .ok_or_else(|| DataFusionError::Execution("Band offset overflow".to_string()))?;

    let ptr = buffer.as_mut_ptr();
    let datatype = <T as GdalType>::datatype().name();
    let mem_path = format!(
        "MEM:::DATAPOINTER={ptr:p},PIXELS={width},LINES={height},BANDS={bands},DATATYPE={datatype},PIXELOFFSET={pixel_offset},LINEOFFSET={line_offset},BANDOFFSET={band_offset}",
        pixel_offset = pixel_size,
    );

    let _guard = ThreadLocalConfigGuard::set(GDAL_MEM_ENABLE_OPEN_KEY, GDAL_MEM_ENABLE_OPEN_VALUE)?;
    let allowed_drivers = ["MEM"];
    Dataset::open_ex(
        &mem_path,
        DatasetOptions {
            open_flags: GdalOpenFlags::GDAL_OF_RASTER
                | GdalOpenFlags::GDAL_OF_UPDATE
                | GdalOpenFlags::GDAL_OF_INTERNAL,
            allowed_drivers: Some(&allowed_drivers),
            ..Default::default()
        },
    )
    .map_err(convert_gdal_err)
}

fn checked_pixels_per_band(width: usize, height: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .ok_or_else(|| DataFusionError::Execution("Pixel count overflow".to_string()))
}

fn checked_pixel_count(width: usize, height: usize, bands: usize) -> Result<usize> {
    checked_pixels_per_band(width, height)?
        .checked_mul(bands)
        .ok_or_else(|| DataFusionError::Execution("Pixel count overflow".to_string()))
}

fn checked_band_index(band_index: usize, bands: usize) -> Result<usize> {
    if band_index == 0 || band_index > bands {
        return Err(DataFusionError::Execution(format!(
            "Band {} is out of range (1-{})",
            band_index, bands
        )));
    }
    Ok(band_index - 1)
}

fn validate_window(
    width: usize,
    height: usize,
    xoff: usize,
    yoff: usize,
    win_w: usize,
    win_h: usize,
) -> Result<()> {
    if win_w == 0 || win_h == 0 {
        return Err(DataFusionError::Execution(
            "Window size must be non-zero".to_string(),
        ));
    }
    if xoff + win_w > width || yoff + win_h > height {
        return Err(DataFusionError::Execution(
            "Window out of bounds".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal::raster::{rasterize, RasterizeOptions};
    use gdal::vector::Geometry;

    #[test]
    fn test_temporary_mem_dataset_read_as_window() {
        let mut dataset = TemporaryMemDataset::<u8>::new_zeroed(4, 3, 1).unwrap();
        let band = dataset.band_slice_mut(1).unwrap();
        for (idx, value) in band.iter_mut().enumerate() {
            *value = idx as u8;
        }

        let window = dataset.read_as(1, (1, 1), (2, 2)).unwrap();
        assert_eq!(window, vec![5, 6, 9, 10]);
    }

    #[test]
    fn test_temporary_mem_dataset_rasterize_second_band() {
        let mut dataset = TemporaryMemDataset::<u8>::new_zeroed(8, 8, 2).unwrap();
        dataset
            .dataset_mut()
            .set_geo_transform(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
            .unwrap();

        let geometry = Geometry::from_wkt("POLYGON ((0 0, 4 0, 4 -4, 0 -4, 0 0))").unwrap();
        rasterize(
            dataset.dataset_mut(),
            &[2],
            &[geometry],
            &[5.0],
            Some(RasterizeOptions::default()),
        )
        .unwrap();

        let band1 = dataset.band_slice(1).unwrap();
        let band2 = dataset.band_slice(2).unwrap();

        assert!(band1.iter().all(|v| *v == 0));
        assert!(band2.iter().any(|v| *v == 5));
    }
}
