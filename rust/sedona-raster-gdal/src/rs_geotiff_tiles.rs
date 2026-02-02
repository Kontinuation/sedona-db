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

//! rs_geotiff_tiles UDTF
//!
//! Read a directory of GeoTIFF files as a table where each row is one internal tile (block)
//! of the source dataset.
//!
//! Output schema:
//! - path: string
//! - x: tile x index (0-based)
//! - y: tile y index (0-based)
//! - rast: out-db raster pointing at the source GeoTIFF band(s)

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{builder::StringBuilder, builder::UInt32Builder, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::TableFunctionImpl;
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PhysicalExpr,
};
use datafusion::{
    common::{plan_err, Result},
    datasource::TableType,
    physical_expr::EquivalenceProperties,
    physical_plan::PlanProperties,
    prelude::Expr,
};
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_common_runtime::SpawnedTask;
use futures::StreamExt;
use gdal::spatial_ref::SpatialRef;
use gdal::{Dataset, DatasetOptions, GdalOpenFlags};
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata};
use sedona_schema::raster::StorageType;
use walkdir::WalkDir;

use crate::gdal_common::{gdal_to_band_data_type, nodata_f64_to_bytes};

/// Create the rs_geotiff_tiles table function
pub fn rs_geotiff_tiles_udtf() -> Arc<dyn TableFunctionImpl> {
    Arc::new(RsGeoTiffTilesFunction {})
}

#[derive(Debug)]
pub struct RsGeoTiffTilesFunction {}

impl TableFunctionImpl for RsGeoTiffTilesFunction {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn datafusion::catalog::TableProvider>> {
        if exprs.is_empty() || exprs.len() > 2 {
            return plan_err!(
                "rs_geotiff_tiles() expected 1 or 2 arguments (dir[, recursive]) but got {}",
                exprs.len()
            );
        }

        let dir = match &exprs[0] {
            Expr::Literal(ScalarValue::Utf8(Some(s)), _) => s.clone(),
            Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => s.to_string(),
            Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => s.clone(),
            other => {
                return plan_err!("rs_geotiff_tiles() expected literal string dir but got {other}")
            }
        };

        let recursive = if exprs.len() == 2 {
            match &exprs[1] {
                Expr::Literal(ScalarValue::Boolean(Some(v)), _) => *v,
                other => {
                    return plan_err!(
                        "rs_geotiff_tiles() expected literal boolean recursive but got {other}"
                    )
                }
            }
        } else {
            false
        };

        Ok(Arc::new(GeoTiffTilesProvider::try_new(dir, recursive)?))
    }
}

#[derive(Debug)]
pub struct GeoTiffTilesProvider {
    dir: String,
    recursive: bool,
    schema: SchemaRef,
}

impl GeoTiffTilesProvider {
    pub fn try_new(dir: String, recursive: bool) -> Result<Self> {
        let rast_field = sedona_schema::datatypes::RASTER
            .to_storage_field("rast", false)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        let schema = Schema::new(vec![
            Field::new("path", DataType::Utf8, false),
            Field::new("x", DataType::UInt32, false),
            Field::new("y", DataType::UInt32, false),
            rast_field,
        ]);

        Ok(Self {
            dir,
            recursive,
            schema: Arc::new(schema),
        })
    }
}

#[async_trait]
impl datafusion::catalog::TableProvider for GeoTiffTilesProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        _state: &dyn datafusion::catalog::Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let exec = Arc::new(GeoTiffTilesExec::new(
            self.dir.clone(),
            self.recursive,
            self.schema.clone(),
        ));

        if let Some(projection) = projection {
            let schema = self.schema();
            let exprs: Vec<_> = projection
                .iter()
                .map(|index| -> (Arc<dyn PhysicalExpr>, String) {
                    let name = schema.field(*index).name();
                    (Arc::new(Column::new(name, *index)), name.clone())
                })
                .collect();
            Ok(Arc::new(ProjectionExec::try_new(exprs, exec)?))
        } else {
            Ok(exec)
        }
    }
}

#[derive(Debug)]
struct GeoTiffTilesExec {
    dir: String,
    recursive: bool,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl GeoTiffTilesExec {
    fn new(dir: String, recursive: bool, schema: SchemaRef) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            dir,
            recursive,
            schema,
            properties,
        }
    }
}

impl DisplayAs for GeoTiffTilesExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "GeoTiffTilesExec: dir='{}', recursive={}",
            self.dir, self.recursive
        )
    }
}

impl ExecutionPlan for GeoTiffTilesExec {
    fn name(&self) -> &str {
        "GeoTiffTilesExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let schema_worker = self.schema.clone();
        let schema_empty = self.schema.clone();
        let schema_adapter = self.schema.clone();
        let dir = self.dir.clone();
        let recursive = self.recursive;

        // Collect paths synchronously
        let paths = list_geotiffs(&dir, recursive)?;

        // Create a stream that processes files in parallel (bounded)
        let stream = futures::stream::iter(paths)
            .map(move |path| {
                let schema = schema_worker.clone();
                SpawnedTask::spawn_blocking(move || build_batch_for_file(path, schema))
            })
            .buffered(4) // Run up to 4 concurrent GDAL opens/reads
            .map(move |res| match res {
                Ok(Ok(Some(batch))) => Ok(batch),
                Ok(Ok(None)) => Ok(RecordBatch::new_empty(schema_empty.clone())),
                Ok(Err(e)) => Err(e),
                Err(e) => Err(DataFusionError::Execution(format!("Task failed: {e}"))),
            });
        //.try_filter(|batch| futures::future::ready(batch.num_rows() > 0));
        // Removed filter to avoid trait imports; empty batches are handled by DataFusion.

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema_adapter,
            Box::pin(stream),
        )))
    }
}

fn build_batch_for_file(path: PathBuf, schema: SchemaRef) -> Result<Option<RecordBatch>> {
    let path_str = path.to_string_lossy().to_string();
    let ds = open_geotiff(&path_str)?;
    let (width, height) = ds.raster_size();

    let band_count = ds.raster_count();
    if band_count == 0 {
        return Ok(None);
    }

    let band1 = ds.rasterband(1).map_err(|e| {
        DataFusionError::Execution(format!("Failed to get band 1 for {path_str}: {e}"))
    })?;
    let (block_x, block_y) = band1.block_size();
    let block_x = block_x.max(1) as u32;
    let block_y = block_y.max(1) as u32;

    let tiles_x = div_ceil_u32(width as u32, block_x);
    let tiles_y = div_ceil_u32(height as u32, block_y);

    let geotransform = ds.geo_transform().map_err(|e| {
        DataFusionError::Execution(format!("Failed to get geotransform for {path_str}: {e}"))
    })?;

    let base_metadata = RasterMetadata {
        width: width as u64,
        height: height as u64,
        upperleft_x: geotransform[0],
        upperleft_y: geotransform[3],
        scale_x: geotransform[1],
        scale_y: geotransform[5],
        skew_x: geotransform[2],
        skew_y: geotransform[4],
    };

    let crs = ds
        .spatial_ref()
        .ok()
        .and_then(|sr: SpatialRef| sr.to_wkt().ok());

    // Estimate capacity
    let total_tiles = (tiles_x * tiles_y) as usize;
    let mut path_builder = StringBuilder::with_capacity(total_tiles, total_tiles * path_str.len());
    let mut x_builder = UInt32Builder::with_capacity(total_tiles);
    let mut y_builder = UInt32Builder::with_capacity(total_tiles);
    let mut rast_builder = RasterBuilder::new(total_tiles);

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            let px = tile_x * block_x;
            let py = tile_y * block_y;

            let tw = (width as u32).saturating_sub(px).min(block_x);
            let th = (height as u32).saturating_sub(py).min(block_y);
            if tw == 0 || th == 0 {
                continue;
            }

            // world coords of this tile's upper-left pixel
            let tile_ulx = base_metadata.upperleft_x
                + (px as f64) * base_metadata.scale_x
                + (py as f64) * base_metadata.skew_x;
            let tile_uly = base_metadata.upperleft_y
                + (px as f64) * base_metadata.skew_y
                + (py as f64) * base_metadata.scale_y;

            let tile_metadata = RasterMetadata {
                width: tw as u64,
                height: th as u64,
                upperleft_x: tile_ulx,
                upperleft_y: tile_uly,
                scale_x: base_metadata.scale_x,
                scale_y: base_metadata.scale_y,
                skew_x: base_metadata.skew_x,
                skew_y: base_metadata.skew_y,
            };

            path_builder.append_value(&path_str);
            x_builder.append_value(tile_x);
            y_builder.append_value(tile_y);

            rast_builder
                .start_raster(&tile_metadata, crs.as_deref())
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to start raster for {path_str} tile ({tile_x},{tile_y}): {e}"
                    ))
                })?;

            for band_idx in 1..=band_count {
                let band = ds.rasterband(band_idx).map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to get band {band_idx} for {path_str}: {e}"
                    ))
                })?;

                let gdal_type = band.band_type();
                let band_data_type = gdal_to_band_data_type(gdal_type).map_err(|_| {
                    DataFusionError::Execution(format!(
                        "Unsupported band data type {gdal_type:?} for {path_str} band {band_idx}"
                    ))
                })?;

                let nodata_bytes = band
                    .no_data_value()
                    .map(|v| nodata_f64_to_bytes(v, &band_data_type));

                let band_metadata = BandMetadata {
                    nodata_value: nodata_bytes,
                    storage_type: StorageType::OutDbRef,
                    datatype: band_data_type,
                    outdb_url: Some(path_str.clone()),
                    outdb_band_id: Some(band_idx as u32),
                };

                rast_builder.start_band(band_metadata).map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to start band {band_idx} for {path_str}: {e}"
                    ))
                })?;

                // Placeholder for out-db pixel bytes.
                rast_builder.band_data_writer().append_value([]);

                rast_builder.finish_band().map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to finish band {band_idx} for {path_str}: {e}"
                    ))
                })?;
            }

            rast_builder.finish_raster().map_err(|e| {
                DataFusionError::Execution(format!(
                    "Failed to finish raster for {path_str} tile ({tile_x},{tile_y}): {e}"
                ))
            })?;
        }
    }

    let rast_array: ArrayRef = Arc::new(
        rast_builder
            .finish()
            .map_err(|e| DataFusionError::Execution(format!("Failed to build rasters: {e}")))?,
    );
    let path_array: ArrayRef = Arc::new(path_builder.finish());
    let x_array: ArrayRef = Arc::new(x_builder.finish());
    let y_array: ArrayRef = Arc::new(y_builder.finish());

    let batch = RecordBatch::try_new(schema, vec![path_array, x_array, y_array, rast_array])
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    Ok(Some(batch))
}

fn list_geotiffs(dir: &str, recursive: bool) -> Result<Vec<PathBuf>> {
    let base = Path::new(dir);
    if !base.exists() {
        return Err(DataFusionError::Execution(format!(
            "rs_geotiff_tiles(): directory does not exist: {dir}"
        )));
    }
    if !base.is_dir() {
        return Err(DataFusionError::Execution(format!(
            "rs_geotiff_tiles(): path is not a directory: {dir}"
        )));
    }

    let mut out = Vec::new();
    let max_depth = if recursive { usize::MAX } else { 1 };
    for entry in WalkDir::new(base)
        .follow_links(false)
        .max_depth(max_depth)
        .into_iter()
    {
        let entry = entry.map_err(|e| DataFusionError::Execution(e.to_string()))?;
        if !entry.file_type().is_file() {
            continue;
        }

        let p = entry.into_path();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());

        if matches!(ext.as_deref(), Some("tif") | Some("tiff")) {
            out.push(p);
        }
    }

    out.sort();
    Ok(out)
}

fn open_geotiff(path: &str) -> Result<Dataset> {
    Dataset::open_ex(
        path,
        DatasetOptions {
            open_flags: GdalOpenFlags::GDAL_OF_RASTER
                | GdalOpenFlags::GDAL_OF_READONLY
                | GdalOpenFlags::GDAL_OF_VERBOSE_ERROR,
            ..Default::default()
        },
    )
    .map_err(|e| DataFusionError::Execution(format!("Failed to open GeoTIFF {path}: {e}")))
}

fn div_ceil_u32(n: u32, d: u32) -> u32 {
    if d == 0 {
        return 0;
    }
    n.div_ceil(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::catalog::TableProvider;
    use datafusion::prelude::SessionContext;
    use sedona_testing::data::test_raster;
    use tempfile::tempdir;

    #[tokio::test]
    async fn udtf_registration_smoke() {
        let ctx = SessionContext::new();
        ctx.register_udtf("rs_geotiff_tiles", rs_geotiff_tiles_udtf());
    }

    #[test]
    fn list_geotiffs_non_recursive() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();
        std::fs::write(base.join("a.tif"), b"not a real tiff").unwrap();
        std::fs::create_dir(base.join("sub")).unwrap();
        std::fs::write(base.join("sub").join("b.tif"), b"not a real tiff").unwrap();

        let files = list_geotiffs(base.to_str().unwrap(), false).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].to_string_lossy().ends_with("a.tif"));
    }

    #[tokio::test]
    async fn provider_builds_rows_for_test_raster() {
        let tmp = tempdir().unwrap();
        let base = tmp.path();

        let src = test_raster("test4.tiff").unwrap();
        let dst = base.join("test4.tiff");
        std::fs::copy(&src, &dst).unwrap();

        let provider = GeoTiffTilesProvider::try_new(base.to_string_lossy().to_string(), false)
            .expect("provider created");

        // Directly call the batch builder to validate schema + non-empty output.
        let batch = build_batch_for_file(dst, provider.schema())
            .expect("build success")
            .expect("batch present");

        assert_eq!(batch.schema().fields().len(), 4);
        assert_eq!(batch.num_columns(), 4);
        // For a 10x10 raster, any reasonable tiling should produce at least one tile.
        assert!(batch.num_rows() >= 1);
    }

    #[test]
    fn rast_field_has_raster_metadata() {
        let provider = GeoTiffTilesProvider::try_new("/tmp".to_string(), false).unwrap();
        let schema = provider.schema();
        let rast_field = schema.field_with_name("rast").unwrap();
        let sedona_type = sedona_schema::datatypes::SedonaType::from_storage_field(rast_field)
            .expect("sedona type");
        assert_eq!(sedona_type, sedona_schema::datatypes::RASTER);
        assert_eq!(
            rast_field
                .metadata()
                .get("ARROW:extension:name")
                .map(|s| s.as_str()),
            Some("sedona.raster")
        );
    }
}
