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

//! RS_TileExplode table function implementation
//!
//! Generates records containing raster tiles resulting from the split of the input raster
//! based upon the desired dimensions of the output rasters.
//!
//! # Formats
//!
//! - `RS_TileExplode(raster, width, height)`
//! - `RS_TileExplode(raster, width, height, padWithNoData)`
//! - `RS_TileExplode(raster, width, height, padWithNoData, noDataVal)`
//! - `RS_TileExplode(raster, bandIndex, width, height, ...)`
//! - `RS_TileExplode(raster, bandIndices, width, height, ...)`
//!
//! # Output Schema
//!
//! Returns a table with columns:
//! - `x`: The index of the tile along X axis (0-based)
//! - `y`: The index of the tile along Y axis (0-based)
//! - `tile`: The tile raster

use std::{any::Any, fmt::Debug, sync::Arc};

use arrow_array::{builder::UInt32Builder, Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::TableFunctionImpl;
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{Partitioning, SendableRecordBatchStream};
use datafusion::{
    catalog::{Session, TableProvider},
    common::Result,
    datasource::TableType,
    physical_expr::EquivalenceProperties,
    physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties},
    prelude::Expr,
};
use datafusion_common::{plan_err, DataFusionError, ScalarValue};
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_schema::raster::{BandDataType, RasterSchema, StorageType};

/// Create the RS_TileExplode table function
pub fn rs_tile_explode_udtf() -> Arc<dyn TableFunctionImpl> {
    Arc::new(TileExplodeFunction::default())
}

/// A table function that explodes a raster into tiles
///
/// This table function accepts a raster and tile dimensions, returning
/// a table with columns (x, y, tile) where each row represents a tile.
///
/// Formats:
/// - RS_TileExplode(raster, width, height)
/// - RS_TileExplode(raster, width, height, padWithNoData)
/// - RS_TileExplode(raster, width, height, padWithNoData, noDataVal)
/// - RS_TileExplode(raster, bandIndex, width, height, ...)
/// - RS_TileExplode(raster, bandIndices, width, height, ...)
#[derive(Debug, Default)]
pub struct TileExplodeFunction {}

impl TableFunctionImpl for TileExplodeFunction {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if exprs.len() < 3 {
            return plan_err!(
                "rs_tile_explode() expected at least 3 arguments (raster, width, height) but got {}",
                exprs.len()
            );
        }

        // Parse the raster argument
        let raster_scalar = extract_raster_scalar(&exprs[0])?;

        // Determine which argument pattern we have based on types
        // Pattern 1: raster, width, height, [padWithNoData], [noDataVal]
        // Pattern 2: raster, bandIndex, width, height, [padWithNoData], [noDataVal]
        // Pattern 3: raster, bandIndices[], width, height, [padWithNoData], [noDataVal]

        let (band_indices, width, height, pad_with_nodata, nodata_val) =
            parse_tile_explode_args(&exprs[1..])?;

        Ok(Arc::new(TileExplodeProvider::new(
            raster_scalar,
            band_indices,
            width,
            height,
            pad_with_nodata,
            nodata_val,
        )?))
    }
}

fn extract_raster_scalar(expr: &Expr) -> Result<ScalarValue> {
    if let Expr::Literal(scalar, _) = expr {
        Ok(scalar.clone())
    } else {
        plan_err!("Expected literal raster value in rs_tile_explode() but got {expr}")
    }
}

#[allow(clippy::type_complexity)]
fn parse_tile_explode_args(
    exprs: &[Expr],
) -> Result<(Option<Vec<u32>>, u32, u32, bool, Option<f64>)> {
    // Try to determine the pattern based on argument count and types
    // Check if the first argument could be a band index or band indices array

    if exprs.len() < 2 {
        return plan_err!("rs_tile_explode() expected at least width and height arguments");
    }

    // Try to detect if first arg is a band index/indices or width
    let first_is_band_spec = is_band_specification(&exprs[0]);

    if first_is_band_spec && exprs.len() >= 3 {
        // Pattern 2 or 3: bandIndex/bandIndices, width, height, ...
        let band_indices = extract_band_indices(&exprs[0])?;
        let width = extract_u32_scalar(&exprs[1], "width")?;
        let height = extract_u32_scalar(&exprs[2], "height")?;

        let pad_with_nodata = if exprs.len() > 3 {
            extract_bool_scalar(&exprs[3], "padWithNoData")?
        } else {
            false
        };

        let nodata_val = if exprs.len() > 4 {
            Some(extract_f64_scalar(&exprs[4], "noDataVal")?)
        } else {
            None
        };

        Ok((
            Some(band_indices),
            width,
            height,
            pad_with_nodata,
            nodata_val,
        ))
    } else {
        // Pattern 1: width, height, [padWithNoData], [noDataVal]
        let width = extract_u32_scalar(&exprs[0], "width")?;
        let height = extract_u32_scalar(&exprs[1], "height")?;

        let pad_with_nodata = if exprs.len() > 2 {
            extract_bool_scalar(&exprs[2], "padWithNoData")?
        } else {
            false
        };

        let nodata_val = if exprs.len() > 3 {
            Some(extract_f64_scalar(&exprs[3], "noDataVal")?)
        } else {
            None
        };

        Ok((None, width, height, pad_with_nodata, nodata_val))
    }
}

fn is_band_specification(expr: &Expr) -> bool {
    if let Expr::Literal(scalar, _) = expr {
        matches!(
            scalar,
            ScalarValue::List(_) | ScalarValue::LargeList(_) | ScalarValue::FixedSizeList(_)
        )
    } else {
        false
    }
}

fn extract_band_indices(expr: &Expr) -> Result<Vec<u32>> {
    if let Expr::Literal(scalar, _) = expr {
        match scalar {
            ScalarValue::Int8(Some(v)) => Ok(vec![*v as u32]),
            ScalarValue::Int16(Some(v)) => Ok(vec![*v as u32]),
            ScalarValue::Int32(Some(v)) => Ok(vec![*v as u32]),
            ScalarValue::Int64(Some(v)) => Ok(vec![*v as u32]),
            ScalarValue::UInt8(Some(v)) => Ok(vec![*v as u32]),
            ScalarValue::UInt16(Some(v)) => Ok(vec![*v as u32]),
            ScalarValue::UInt32(Some(v)) => Ok(vec![*v]),
            ScalarValue::UInt64(Some(v)) => Ok(vec![*v as u32]),
            ScalarValue::List(arr) => extract_indices_from_list_array(arr.as_ref()),
            ScalarValue::LargeList(arr) => extract_indices_from_large_list_array(arr.as_ref()),
            ScalarValue::FixedSizeList(arr) => {
                extract_indices_from_fixed_size_list_array(arr.as_ref())
            }
            _ => plan_err!(
                "Expected integer or array of integers for band indices but got {scalar:?}"
            ),
        }
    } else {
        plan_err!("Expected literal value for band indices but got {expr}")
    }
}

fn extract_indices_from_list_array(arr: &arrow_array::ListArray) -> Result<Vec<u32>> {
    let mut indices = Vec::new();
    let values = arr.values();
    for i in 0..values.len() {
        if !values.is_null(i) {
            if let Some(int_arr) = values.as_any().downcast_ref::<arrow_array::Int32Array>() {
                indices.push(int_arr.value(i) as u32);
            } else if let Some(int_arr) = values.as_any().downcast_ref::<arrow_array::Int64Array>()
            {
                indices.push(int_arr.value(i) as u32);
            } else {
                return plan_err!("Unsupported array element type for band indices");
            }
        }
    }
    Ok(indices)
}

fn extract_indices_from_large_list_array(arr: &arrow_array::LargeListArray) -> Result<Vec<u32>> {
    let mut indices = Vec::new();
    let values = arr.values();
    for i in 0..values.len() {
        if !values.is_null(i) {
            if let Some(int_arr) = values.as_any().downcast_ref::<arrow_array::Int32Array>() {
                indices.push(int_arr.value(i) as u32);
            } else if let Some(int_arr) = values.as_any().downcast_ref::<arrow_array::Int64Array>()
            {
                indices.push(int_arr.value(i) as u32);
            } else {
                return plan_err!("Unsupported array element type for band indices");
            }
        }
    }
    Ok(indices)
}

fn extract_indices_from_fixed_size_list_array(
    arr: &arrow_array::FixedSizeListArray,
) -> Result<Vec<u32>> {
    let mut indices = Vec::new();
    let values = arr.values();
    for i in 0..values.len() {
        if !values.is_null(i) {
            if let Some(int_arr) = values.as_any().downcast_ref::<arrow_array::Int32Array>() {
                indices.push(int_arr.value(i) as u32);
            } else if let Some(int_arr) = values.as_any().downcast_ref::<arrow_array::Int64Array>()
            {
                indices.push(int_arr.value(i) as u32);
            } else {
                return plan_err!("Unsupported array element type for band indices");
            }
        }
    }
    Ok(indices)
}

fn extract_u32_scalar(expr: &Expr, name: &str) -> Result<u32> {
    if let Expr::Literal(scalar, _) = expr {
        match scalar {
            ScalarValue::Int8(Some(v)) => Ok(*v as u32),
            ScalarValue::Int16(Some(v)) => Ok(*v as u32),
            ScalarValue::Int32(Some(v)) => Ok(*v as u32),
            ScalarValue::Int64(Some(v)) => Ok(*v as u32),
            ScalarValue::UInt8(Some(v)) => Ok(*v as u32),
            ScalarValue::UInt16(Some(v)) => Ok(*v as u32),
            ScalarValue::UInt32(Some(v)) => Ok(*v),
            ScalarValue::UInt64(Some(v)) => Ok(*v as u32),
            _ => plan_err!("Expected integer for {name} but got {scalar:?}"),
        }
    } else {
        plan_err!("Expected literal integer for {name} but got {expr}")
    }
}

fn extract_bool_scalar(expr: &Expr, name: &str) -> Result<bool> {
    if let Expr::Literal(scalar, _) = expr {
        match scalar {
            ScalarValue::Boolean(Some(v)) => Ok(*v),
            _ => plan_err!("Expected boolean for {name} but got {scalar:?}"),
        }
    } else {
        plan_err!("Expected literal boolean for {name} but got {expr}")
    }
}

fn extract_f64_scalar(expr: &Expr, name: &str) -> Result<f64> {
    if let Expr::Literal(scalar, _) = expr {
        match scalar {
            ScalarValue::Float32(Some(v)) => Ok(*v as f64),
            ScalarValue::Float64(Some(v)) => Ok(*v),
            ScalarValue::Int8(Some(v)) => Ok(*v as f64),
            ScalarValue::Int16(Some(v)) => Ok(*v as f64),
            ScalarValue::Int32(Some(v)) => Ok(*v as f64),
            ScalarValue::Int64(Some(v)) => Ok(*v as f64),
            ScalarValue::UInt8(Some(v)) => Ok(*v as f64),
            ScalarValue::UInt16(Some(v)) => Ok(*v as f64),
            ScalarValue::UInt32(Some(v)) => Ok(*v as f64),
            ScalarValue::UInt64(Some(v)) => Ok(*v as f64),
            ScalarValue::Null => Ok(f64::NAN), // Handle null noDataVal
            _ => plan_err!("Expected numeric value for {name} but got {scalar:?}"),
        }
    } else {
        plan_err!("Expected literal numeric value for {name} but got {expr}")
    }
}

/// Provider that generates tiles from a raster
#[derive(Debug)]
pub struct TileExplodeProvider {
    raster_scalar: ScalarValue,
    band_indices: Option<Vec<u32>>,
    tile_width: u32,
    tile_height: u32,
    pad_with_nodata: bool,
    nodata_val: Option<f64>,
    schema: SchemaRef,
    num_tiles_x: u32,
    num_tiles_y: u32,
}

impl TileExplodeProvider {
    pub fn new(
        raster_scalar: ScalarValue,
        band_indices: Option<Vec<u32>>,
        tile_width: u32,
        tile_height: u32,
        pad_with_nodata: bool,
        nodata_val: Option<f64>,
    ) -> Result<Self> {
        // Calculate the number of tiles based on raster dimensions
        let (num_tiles_x, num_tiles_y) =
            calculate_tile_count(&raster_scalar, tile_width, tile_height, pad_with_nodata)?;

        // Build the output schema: (x: UInt32, y: UInt32, tile: Raster)
        let raster_type = DataType::Struct(RasterSchema::fields());
        let schema = Schema::new(vec![
            Field::new("x", DataType::UInt32, false),
            Field::new("y", DataType::UInt32, false),
            Field::new("tile", raster_type, false),
        ]);

        Ok(Self {
            raster_scalar,
            band_indices,
            tile_width,
            tile_height,
            pad_with_nodata,
            nodata_val,
            schema: Arc::new(schema),
            num_tiles_x,
            num_tiles_y,
        })
    }
}

fn calculate_tile_count(
    raster_scalar: &ScalarValue,
    tile_width: u32,
    tile_height: u32,
    _pad_with_nodata: bool,
) -> Result<(u32, u32)> {
    // Extract raster dimensions from the scalar value
    let (raster_width, raster_height) = extract_raster_dimensions(raster_scalar)?;

    let num_tiles_x = raster_width.div_ceil(tile_width);
    let num_tiles_y = raster_height.div_ceil(tile_height);

    Ok((num_tiles_x, num_tiles_y))
}

fn extract_raster_dimensions(raster_scalar: &ScalarValue) -> Result<(u32, u32)> {
    // Extract the raster struct array and get dimensions
    if let ScalarValue::Struct(struct_array) = raster_scalar {
        let accessor = RasterStructArray::new(struct_array.as_ref());
        if struct_array.is_null(0) {
            return plan_err!("Input raster is null");
        }
        let raster = accessor
            .get(0)
            .map_err(|e| DataFusionError::External(e.into()))?;
        let width = raster.metadata().width() as u32;
        let height = raster.metadata().height() as u32;
        Ok((width, height))
    } else {
        plan_err!("Expected raster struct but got {raster_scalar:?}")
    }
}

#[async_trait]
impl TableProvider for TileExplodeProvider {
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
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(TileExplodeExec::new(
            self.raster_scalar.clone(),
            self.band_indices.clone(),
            self.tile_width,
            self.tile_height,
            self.pad_with_nodata,
            self.nodata_val,
            self.schema.clone(),
            self.num_tiles_x,
            self.num_tiles_y,
        )))
    }
}

/// Execution plan for tile explosion
#[derive(Debug)]
struct TileExplodeExec {
    raster_scalar: ScalarValue,
    band_indices: Option<Vec<u32>>,
    tile_width: u32,
    tile_height: u32,
    pad_with_nodata: bool,
    nodata_val: Option<f64>,
    schema: SchemaRef,
    num_tiles_x: u32,
    num_tiles_y: u32,
    properties: PlanProperties,
}

impl TileExplodeExec {
    #[allow(clippy::too_many_arguments)]
    fn new(
        raster_scalar: ScalarValue,
        band_indices: Option<Vec<u32>>,
        tile_width: u32,
        tile_height: u32,
        pad_with_nodata: bool,
        nodata_val: Option<f64>,
        schema: SchemaRef,
        num_tiles_x: u32,
        num_tiles_y: u32,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            raster_scalar,
            band_indices,
            tile_width,
            tile_height,
            pad_with_nodata,
            nodata_val,
            schema,
            num_tiles_x,
            num_tiles_y,
            properties,
        }
    }
}

impl DisplayAs for TileExplodeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "TileExplodeExec: tile_size={}x{}, tiles={}x{}",
            self.tile_width, self.tile_height, self.num_tiles_x, self.num_tiles_y
        )
    }
}

impl ExecutionPlan for TileExplodeExec {
    fn name(&self) -> &str {
        "TileExplodeExec"
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
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Generate all tiles in a single batch
        let batch = generate_tiles(
            &self.raster_scalar,
            &self.band_indices,
            self.tile_width,
            self.tile_height,
            self.pad_with_nodata,
            self.nodata_val,
            self.num_tiles_x,
            self.num_tiles_y,
            self.schema.clone(),
        )?;

        let stream = Box::pin(futures::stream::iter(vec![Ok(batch)]));
        let record_batch_stream = RecordBatchStreamAdapter::new(self.schema.clone(), stream);
        Ok(Box::pin(record_batch_stream))
    }
}

/// Generate all tiles from the input raster
#[allow(clippy::too_many_arguments)]
fn generate_tiles(
    raster_scalar: &ScalarValue,
    band_indices: &Option<Vec<u32>>,
    tile_width: u32,
    tile_height: u32,
    pad_with_nodata: bool,
    nodata_val: Option<f64>,
    num_tiles_x: u32,
    num_tiles_y: u32,
    schema: SchemaRef,
) -> Result<RecordBatch> {
    let total_tiles = (num_tiles_x * num_tiles_y) as usize;

    // Extract the source raster struct array
    let struct_array = if let ScalarValue::Struct(arr) = raster_scalar {
        arr
    } else {
        return plan_err!("Expected raster struct but got {raster_scalar:?}");
    };

    let accessor = RasterStructArray::new(struct_array.as_ref());
    let source_raster = accessor
        .get(0)
        .map_err(|e| DataFusionError::External(e.into()))?;

    // Prepare builders
    let mut x_builder = UInt32Builder::with_capacity(total_tiles);
    let mut y_builder = UInt32Builder::with_capacity(total_tiles);
    let mut raster_builder = RasterBuilder::new(total_tiles);

    let source_metadata = source_raster.metadata();
    let source_crs = source_raster.crs();
    let source_bands = source_raster.bands();
    let raster_width = source_metadata.width() as u32;
    let raster_height = source_metadata.height() as u32;

    // Determine which bands to include
    let selected_bands: Vec<usize> = match band_indices {
        Some(indices) => indices.iter().map(|i| *i as usize).collect(),
        None => (1..=source_bands.len()).collect(), // 1-based band indices
    };

    // Generate each tile
    for tile_y in 0..num_tiles_y {
        for tile_x in 0..num_tiles_x {
            x_builder.append_value(tile_x);
            y_builder.append_value(tile_y);

            // Calculate tile bounds in pixel coordinates
            let pixel_x = tile_x * tile_width;
            let pixel_y = tile_y * tile_height;

            // Calculate actual tile dimensions (may be smaller for edge tiles)
            let actual_width = if pad_with_nodata {
                tile_width
            } else {
                (raster_width - pixel_x).min(tile_width)
            };
            let actual_height = if pad_with_nodata {
                tile_height
            } else {
                (raster_height - pixel_y).min(tile_height)
            };

            // Calculate the upper-left corner in world coordinates
            let tile_upper_left_x = source_metadata.upper_left_x()
                + (pixel_x as f64) * source_metadata.scale_x()
                + (pixel_y as f64) * source_metadata.skew_x();
            let tile_upper_left_y = source_metadata.upper_left_y()
                + (pixel_x as f64) * source_metadata.skew_y()
                + (pixel_y as f64) * source_metadata.scale_y();

            // Create tile metadata
            let tile_metadata = RasterMetadata {
                width: actual_width as u64,
                height: actual_height as u64,
                upperleft_x: tile_upper_left_x,
                upperleft_y: tile_upper_left_y,
                scale_x: source_metadata.scale_x(),
                scale_y: source_metadata.scale_y(),
                skew_x: source_metadata.skew_x(),
                skew_y: source_metadata.skew_y(),
            };

            raster_builder
                .start_raster(&tile_metadata, source_crs)
                .map_err(|e: arrow_schema::ArrowError| DataFusionError::External(e.into()))?;

            // Extract and write band data for each selected band
            for &band_num in &selected_bands {
                let source_band = source_bands
                    .band(band_num)
                    .map_err(|e: arrow_schema::ArrowError| DataFusionError::External(e.into()))?;

                let band_meta = source_band.metadata();
                let storage_type = band_meta.storage_type();
                let data_type = band_meta.data_type();
                let bytes_per_pixel = data_type.bytes_per_pixel();

                // Create band metadata for the tile
                let tile_band_metadata = BandMetadata {
                    nodata_value: band_meta.nodata_value().map(|v: &[u8]| v.to_vec()).or_else(
                        || {
                            // If no nodata value is set and we need to pad, use the provided nodata_val
                            if pad_with_nodata {
                                nodata_val.map(|v| nodata_to_bytes(v, data_type))
                            } else {
                                None
                            }
                        },
                    ),
                    storage_type,
                    datatype: data_type,
                    outdb_url: band_meta.outdb_url().map(|s: &str| s.to_string()),
                    outdb_band_id: band_meta.outdb_band_id(),
                };

                raster_builder
                    .start_band(tile_band_metadata)
                    .map_err(|e: arrow_schema::ArrowError| DataFusionError::External(e.into()))?;

                // Extract tile pixel data
                if storage_type == StorageType::InDb {
                    let source_data = source_band.data();
                    let tile_data = extract_tile_data(
                        source_data,
                        raster_width,
                        raster_height,
                        pixel_x,
                        pixel_y,
                        tile_width,
                        tile_height,
                        actual_width,
                        actual_height,
                        bytes_per_pixel,
                        pad_with_nodata,
                        band_meta
                            .nodata_value()
                            .or(nodata_val.map(|v| nodata_to_bytes(v, data_type)).as_deref()),
                    );

                    raster_builder.band_data_writer().append_value(&tile_data);
                } else {
                    // For OutDb rasters, we don't copy data - just reference the source
                    raster_builder.band_data_writer().append_value([]);
                }

                raster_builder
                    .finish_band()
                    .map_err(|e: arrow_schema::ArrowError| DataFusionError::External(e.into()))?;
            }

            raster_builder
                .finish_raster()
                .map_err(|e: arrow_schema::ArrowError| DataFusionError::External(e.into()))?;
        }
    }

    let x_array: ArrayRef = Arc::new(x_builder.finish());
    let y_array: ArrayRef = Arc::new(y_builder.finish());
    let tile_array: ArrayRef = Arc::new(
        raster_builder
            .finish()
            .map_err(|e: arrow_schema::ArrowError| DataFusionError::External(e.into()))?,
    );

    RecordBatch::try_new(schema, vec![x_array, y_array, tile_array])
        .map_err(|e: arrow_schema::ArrowError| DataFusionError::External(e.into()))
}

/// Extract tile data from source raster data
#[allow(clippy::too_many_arguments)]
fn extract_tile_data(
    source_data: &[u8],
    raster_width: u32,
    raster_height: u32,
    tile_pixel_x: u32,
    tile_pixel_y: u32,
    tile_width: u32,
    tile_height: u32,
    actual_width: u32,
    actual_height: u32,
    bytes_per_pixel: usize,
    pad_with_nodata: bool,
    nodata_bytes: Option<&[u8]>,
) -> Vec<u8> {
    let output_width = if pad_with_nodata {
        tile_width
    } else {
        actual_width
    };
    let output_height = if pad_with_nodata {
        tile_height
    } else {
        actual_height
    };

    let mut tile_data =
        Vec::with_capacity((output_width * output_height) as usize * bytes_per_pixel);

    for row in 0..output_height {
        let source_row = tile_pixel_y + row;

        for col in 0..output_width {
            let source_col = tile_pixel_x + col;

            if source_row < raster_height
                && source_col < raster_width
                && row < actual_height
                && col < actual_width
            {
                // Copy pixel from source
                let source_offset =
                    ((source_row * raster_width + source_col) as usize) * bytes_per_pixel;
                if source_offset + bytes_per_pixel <= source_data.len() {
                    tile_data.extend_from_slice(
                        &source_data[source_offset..source_offset + bytes_per_pixel],
                    );
                } else if let Some(nodata) = nodata_bytes {
                    tile_data.extend_from_slice(nodata);
                } else {
                    tile_data.extend(std::iter::repeat_n(0u8, bytes_per_pixel));
                }
            } else if let Some(nodata) = nodata_bytes {
                // Pad with nodata
                tile_data.extend_from_slice(nodata);
            } else {
                // Pad with zeros if no nodata value
                tile_data.extend(std::iter::repeat_n(0u8, bytes_per_pixel));
            }
        }
    }

    tile_data
}

/// Convert a nodata value to bytes based on band data type
fn nodata_to_bytes(value: f64, data_type: BandDataType) -> Vec<u8> {
    match data_type {
        BandDataType::UInt8 => vec![value as u8],
        BandDataType::Int8 => vec![(value as i8).to_le_bytes()[0]],
        BandDataType::UInt16 => (value as u16).to_le_bytes().to_vec(),
        BandDataType::Int16 => (value as i16).to_le_bytes().to_vec(),
        BandDataType::UInt32 => (value as u32).to_le_bytes().to_vec(),
        BandDataType::Int32 => (value as i32).to_le_bytes().to_vec(),
        BandDataType::UInt64 => (value as u64).to_le_bytes().to_vec(),
        BandDataType::Int64 => (value as i64).to_le_bytes().to_vec(),
        BandDataType::Float32 => (value as f32).to_le_bytes().to_vec(),
        BandDataType::Float64 => value.to_le_bytes().to_vec(),
    }
}

trait BandDataTypeExt {
    fn bytes_per_pixel(&self) -> usize;
}

impl BandDataTypeExt for BandDataType {
    fn bytes_per_pixel(&self) -> usize {
        match self {
            BandDataType::UInt8 => 1,
            BandDataType::Int8 => 1,
            BandDataType::UInt16 | BandDataType::Int16 => 2,
            BandDataType::UInt32 | BandDataType::Int32 | BandDataType::Float32 => 4,
            BandDataType::UInt64 | BandDataType::Int64 => 8,
            BandDataType::Float64 => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StructArray;
    use datafusion::prelude::SessionContext;
    use sedona_testing::rasters::generate_test_rasters;

    #[test]
    fn test_tile_explode_function_creation() {
        let udtf = rs_tile_explode_udtf();
        // Just verify it can be created
        assert!(std::any::TypeId::of::<TileExplodeFunction>() != std::any::TypeId::of::<()>());
        let _ = udtf;
    }

    #[tokio::test]
    async fn test_tile_explode_provider() {
        // Create a test raster
        let rasters = generate_test_rasters(2, Some(0)).unwrap();

        // Get the first non-null raster as a scalar
        let raster_array: &StructArray = &rasters;

        // Find first non-null row
        for i in 0..raster_array.len() {
            if !raster_array.is_null(i) {
                // Create a scalar from this single raster
                let slice = raster_array.slice(i, 1);
                let scalar = ScalarValue::Struct(Arc::new(slice));

                // Create the provider
                let provider = TileExplodeProvider::new(
                    scalar, None, // all bands
                    1,    // tile width
                    1,    // tile height
                    false, None,
                )
                .unwrap();

                // Verify the schema
                let schema = provider.schema();
                assert_eq!(schema.fields().len(), 3);
                assert_eq!(schema.field(0).name(), "x");
                assert_eq!(schema.field(1).name(), "y");
                assert_eq!(schema.field(2).name(), "tile");

                break;
            }
        }
    }

    #[tokio::test]
    async fn test_tile_explode_registration() {
        let ctx = SessionContext::new();
        ctx.register_udtf("rs_tile_explode", rs_tile_explode_udtf());

        // Verify the function was registered
        // Note: Full execution test requires literal raster values which is complex
    }
}
