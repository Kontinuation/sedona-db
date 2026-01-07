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

use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    Array, AsArray, Float64Array, ListArray, RecordBatch, StructArray, UInt64Array,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::interleave_record_batch;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow_select::interleave::interleave as arrow_interleave;
use datafusion::config::SpillCompression;
use datafusion_common::Result;
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_plan::metrics::SpillMetrics;
use parking_lot::Mutex;
use sedona_common::sedona_internal_err;

use crate::utils::arrow_utils::compact_batch;
use crate::utils::spill::{RecordBatchSpillReader, RecordBatchSpillWriter};

fn used_source_mask(len: usize, indices: &[(usize, usize)]) -> Vec<bool> {
    let mut used = vec![false; len];
    for (src_i, _) in indices {
        debug_assert!(*src_i < len);
        used[*src_i] = true;
    }
    used
}

/// KNNResultsMerger handles the merging of KNN "nearest so far" results from multiple partitions.
/// It maintains spill files to store intermediate results.
pub struct KNNResultsMerger {
    k: usize,
    include_tie_breaker: bool,
    target_batch_size: usize,
    target_spilled_batch_size: usize,
    /// Schema of the final result (without distance)
    result_schema: SchemaRef,
    /// Schema for the intermediate spill files
    spill_schema: SchemaRef,
    /// Runtime env
    runtime_env: Arc<RuntimeEnv>,
    /// Spill compression
    spill_compression: SpillCompression,
    /// Spill metrics
    spill_metrics: SpillMetrics,
    /// State protected by mutex
    state: Mutex<MergerState>,
}

struct MergerState {
    /// File containing results from previous (0..N-1) partitions
    previous_file: Option<RefCountedTempFile>,
    /// Reader for previous file
    previous_reader: Option<SpillRowReader>,
    /// Spill writer for current (0..N) partitions
    current_writer: Option<RecordBatchSpillWriter>,

    /// When the upstream batching (`max_batch_size`) splits a single probe row across
    /// multiple `ingest` calls, we must not emit the same probe index multiple times.
    ///
    /// We buffer the currently-being-accumulated probe row here until we are confident
    /// it's complete for the current indexed partition.
    pending: PendingProbe,

    /// Builds spill rows (probe-index rows) without materializing a RecordBatch per probe index.
    spill_builder: SpillRowBuilder,

    /// Stages spill batches (from spill_builder and gap-fill carry-over) and flushes using
    /// Arrow interleave instead of concat.
    spill_stage: SpillBatchStage,

    /// Builds final output rows and emits batches using record-batch interleave.
    output_builder: OutputRowBuilder,
}

/// FIFO staging area for spill batches that flushes using Arrow record-batch interleave.
///
/// This replaces `concat_batches` on a potentially huge list of tiny batches.
struct SpillBatchStage {
    batches: VecDeque<RecordBatch>,
    num_rows: usize,
}

impl SpillBatchStage {
    fn new() -> Self {
        Self {
            batches: VecDeque::new(),
            num_rows: 0,
        }
    }

    fn push_batch(&mut self, batch: RecordBatch, target_rows: usize) {
        let n = batch.num_rows();
        if n == 0 {
            return;
        }

        if n <= target_rows {
            self.num_rows += n;
            self.batches.push_back(batch);
            return;
        }

        // Split oversized batches so the flush loop can always assemble ~target_rows chunks.
        let mut start = 0;
        while start < n {
            let len = target_rows.min(n - start);
            self.num_rows += len;
            self.batches.push_back(batch.slice(start, len));
            start += len;
        }
    }

    fn maybe_flush(
        &mut self,
        writer: &mut RecordBatchSpillWriter,
        spill_schema: &SchemaRef,
        target_rows: usize,
    ) -> Result<()> {
        while self.num_rows >= target_rows {
            let chunk = self.take_chunk(target_rows);
            let batch = interleave_concat_record_batches(spill_schema, &chunk)?;
            let batch = compact_batch(batch)?;
            writer.write_batch(&batch)?;
        }
        Ok(())
    }

    fn flush_all(
        &mut self,
        writer: &mut RecordBatchSpillWriter,
        spill_schema: &SchemaRef,
        target_rows: usize,
    ) -> Result<()> {
        while self.num_rows > 0 {
            let take = self.num_rows.min(target_rows.max(1));
            let chunk = self.take_chunk(take);
            let batch = interleave_concat_record_batches(spill_schema, &chunk)?;
            let batch = compact_batch(batch)?;
            writer.write_batch(&batch)?;
        }
        Ok(())
    }

    fn take_chunk(&mut self, target_rows: usize) -> Vec<RecordBatch> {
        debug_assert!(target_rows > 0);
        debug_assert!(self.num_rows >= target_rows);

        let mut remaining = target_rows;
        let mut out: Vec<RecordBatch> = Vec::new();

        while remaining > 0 {
            let batch = self
                .batches
                .pop_front()
                .expect("spill stage should not be empty");
            let n = batch.num_rows();

            if n <= remaining {
                out.push(batch);
                self.num_rows -= n;
                remaining -= n;
            } else {
                let head = batch.slice(0, remaining);
                let tail = batch.slice(remaining, n - remaining);
                out.push(head);
                self.batches.push_front(tail);
                self.num_rows -= remaining;
                remaining = 0;
            }
        }

        out
    }
}

fn interleave_concat_record_batches(
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<RecordBatch> {
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(schema.clone()));
    }
    if batches.len() == 1 {
        return Ok(batches[0].clone());
    }

    let refs: Vec<&RecordBatch> = batches.iter().collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let mut indices: Vec<(usize, usize)> = Vec::with_capacity(total_rows);
    for (bi, b) in batches.iter().enumerate() {
        for ri in 0..b.num_rows() {
            indices.push((bi, ri));
        }
    }
    let batch = interleave_record_batch(&refs, &indices)?;
    debug_assert_eq!(batch.num_rows(), total_rows);
    debug_assert_eq!(batch.schema().as_ref(), schema.as_ref());
    Ok(batch)
}

/// Builder for spill rows (one row per probe index) that materializes a spill RecordBatch in one go.
///
/// Uses array interleave for the nested `row` StructArray to avoid scalar materialization.
struct SpillRowBuilder {
    probe_indices: Vec<u64>,

    // rows: List<Struct<row, dist>>
    rows_offsets: Vec<i32>,
    row_sources: Vec<Arc<StructArray>>,
    row_source_map: HashMap<usize, usize>,
    rows_value_indices: Vec<(usize, usize)>,
    rows_dist_values: Vec<f64>,

    // unfiltered_dists: List<Float64>
    unfiltered_offsets: Vec<i32>,
    unfiltered_values: Vec<f64>,
}

impl SpillRowBuilder {
    fn split_prefix<T>(values: &mut Vec<T>, take: usize) -> Vec<T> {
        let remaining = values.split_off(take);
        std::mem::replace(values, remaining)
    }

    fn split_list_offsets(offsets: &mut Vec<i32>, take_rows: usize) -> Vec<i32> {
        debug_assert!(take_rows < offsets.len());
        let taken = offsets[..=take_rows].to_vec();
        let base = offsets[take_rows];
        *offsets = offsets[take_rows..].iter().map(|o| *o - base).collect();
        taken
    }

    fn new() -> Self {
        Self {
            probe_indices: Vec::new(),
            rows_offsets: vec![0],
            row_sources: Vec::new(),
            row_source_map: HashMap::new(),
            rows_value_indices: Vec::new(),
            rows_dist_values: Vec::new(),
            unfiltered_offsets: vec![0],
            unfiltered_values: Vec::new(),
        }
    }

    fn len_rows(&self) -> usize {
        self.probe_indices.len()
    }

    fn is_empty(&self) -> bool {
        self.probe_indices.is_empty()
    }

    fn reset(&mut self) {
        self.probe_indices.clear();
        self.rows_offsets.clear();
        self.rows_offsets.push(0);
        self.row_sources.clear();
        self.row_source_map.clear();
        self.rows_value_indices.clear();
        self.rows_dist_values.clear();
        self.unfiltered_offsets.clear();
        self.unfiltered_offsets.push(0);
        self.unfiltered_values.clear();
    }

    fn register_row_source(&mut self, source: &Arc<StructArray>) -> usize {
        let key = Arc::as_ptr(source) as usize;
        if let Some(existing) = self.row_source_map.get(&key) {
            return *existing;
        }
        let idx = self.row_sources.len();
        self.row_sources.push(source.clone());
        self.row_source_map.insert(key, idx);
        idx
    }

    fn compact_row_sources(&mut self) {
        if self.rows_value_indices.is_empty() {
            self.row_sources.clear();
            self.row_source_map.clear();
            return;
        }

        let used = used_source_mask(self.row_sources.len(), &self.rows_value_indices);

        // Old source index -> new source index.
        let mut remap = vec![usize::MAX; self.row_sources.len()];
        let mut new_sources = Vec::with_capacity(self.row_sources.len());
        let mut new_map = HashMap::with_capacity(self.row_source_map.len());

        for (old_i, src) in self.row_sources.iter().enumerate() {
            if !used[old_i] {
                continue;
            }

            let new_i = new_sources.len();
            new_sources.push(src.clone());
            remap[old_i] = new_i;
            let key = Arc::as_ptr(src) as usize;
            new_map.insert(key, new_i);
        }

        for (src_i, _) in &mut self.rows_value_indices {
            *src_i = remap[*src_i];
            debug_assert_ne!(*src_i, usize::MAX);
        }

        self.row_sources = new_sources;
        self.row_source_map = new_map;
    }

    fn push_row(
        &mut self,
        idx: usize,
        candidates: &[Candidate],
        unfiltered_dists: &[f64],
    ) -> Result<()> {
        self.probe_indices.push(idx as u64);

        let prev = *self
            .rows_offsets
            .last()
            .expect("rows_offsets must have at least one element") as usize;
        let new_total = prev + candidates.len();
        let new_total_i32: i32 = if new_total <= i32::MAX as usize {
            new_total as i32
        } else {
            return sedona_internal_err!("spill rows list too large");
        };
        self.rows_offsets.push(new_total_i32);

        self.rows_value_indices.reserve(candidates.len());
        self.rows_dist_values.reserve(candidates.len());
        for c in candidates {
            let src_i = self.register_row_source(&c.row.data);
            self.rows_value_indices.push((src_i, c.row.row));
            self.rows_dist_values.push(c.dist);
        }

        let prev_u = *self
            .unfiltered_offsets
            .last()
            .expect("unfiltered_offsets must have at least one element")
            as usize;
        let new_u_total = prev_u + unfiltered_dists.len();
        let new_u_total_i32: i32 = if new_u_total <= i32::MAX as usize {
            new_u_total as i32
        } else {
            return sedona_internal_err!("spill unfiltered list too large");
        };
        self.unfiltered_offsets.push(new_u_total_i32);
        self.unfiltered_values.extend_from_slice(unfiltered_dists);

        Ok(())
    }

    fn take_batch(
        &mut self,
        spill_schema: &SchemaRef,
        result_schema: &SchemaRef,
        max_rows: usize,
    ) -> Result<Option<RecordBatch>> {
        if self.probe_indices.is_empty() {
            return Ok(None);
        }
        let take_rows = self.probe_indices.len().min(max_rows.max(1));

        // Determine value ranges for the two list columns.
        let rows_values_end = self.rows_offsets[take_rows] as usize;
        let unfiltered_values_end = self.unfiltered_offsets[take_rows] as usize;

        // Split the row-wise vectors.
        let take_indices = Self::split_prefix(&mut self.probe_indices, take_rows);

        // Split list offsets and normalize remaining offsets to start at 0.
        let take_rows_offsets = Self::split_list_offsets(&mut self.rows_offsets, take_rows);
        let take_unfiltered_offsets =
            Self::split_list_offsets(&mut self.unfiltered_offsets, take_rows);

        // Split value vectors.
        let take_rows_value_indices =
            Self::split_prefix(&mut self.rows_value_indices, rows_values_end);
        let take_rows_dist_values = Self::split_prefix(&mut self.rows_dist_values, rows_values_end);
        let take_unfiltered_values =
            Self::split_prefix(&mut self.unfiltered_values, unfiltered_values_end);

        // Build the spill record batch for the taken prefix.
        let batch = build_spill_batch_from_parts(
            spill_schema,
            result_schema,
            &take_indices,
            &self.row_sources,
            &take_rows_value_indices,
            &take_rows_dist_values,
            &take_rows_offsets,
            &take_unfiltered_values,
            &take_unfiltered_offsets,
        )?;

        // Drop any unreferenced sources as early as possible, so memory is bounded by the
        // remaining staged indices (not by historical sources encountered).
        self.compact_row_sources();
        Ok(Some(batch))
    }
}

fn build_spill_batch_from_parts(
    spill_schema: &SchemaRef,
    result_schema: &SchemaRef,
    probe_indices: &[u64],
    row_sources: &[Arc<StructArray>],
    rows_value_indices: &[(usize, usize)],
    rows_dist_values: &[f64],
    rows_offsets: &[i32],
    unfiltered_values: &[f64],
    unfiltered_offsets: &[i32],
) -> Result<RecordBatch> {
    // index column
    let index_arr = UInt64Array::from(probe_indices.to_vec());

    // rows column: List<Struct<row: Struct<result_schema>, dist: Float64>>
    let row_arrays: Vec<&dyn Array> = row_sources
        .iter()
        .map(|s| s.as_ref() as &dyn Array)
        .collect();
    let interleaved_rows: Arc<dyn Array> = if rows_value_indices.is_empty() {
        arrow::array::new_empty_array(&DataType::Struct(result_schema.fields().clone()))
    } else {
        arrow_interleave(&row_arrays, rows_value_indices)?
    };

    let dist_arr = Float64Array::from(rows_dist_values.to_vec());

    let row_field = Field::new(
        "row",
        DataType::Struct(result_schema.fields().clone()),
        false,
    );
    let dist_field = Field::new("dist", DataType::Float64, false);
    let value_fields = vec![row_field, dist_field];

    let values_struct = StructArray::try_new(
        value_fields.into(),
        vec![interleaved_rows, Arc::new(dist_arr)],
        None,
    )?;

    let list_offsets = OffsetBuffer::<i32>::new(rows_offsets.to_vec().into());
    let list_field = Arc::new(Field::new("item", values_struct.data_type().clone(), true));
    let rows_list = ListArray::try_new(list_field, list_offsets, Arc::new(values_struct), None)?;

    // unfiltered_dists column: List<Float64>
    let unfiltered_values_arr = Float64Array::from(unfiltered_values.to_vec());
    let unfiltered_offsets_buf = OffsetBuffer::<i32>::new(unfiltered_offsets.to_vec().into());
    let unfiltered_field = Arc::new(Field::new("item", DataType::Float64, true));
    let unfiltered_list = ListArray::try_new(
        unfiltered_field,
        unfiltered_offsets_buf,
        Arc::new(unfiltered_values_arr),
        None,
    )?;

    Ok(RecordBatch::try_new(
        spill_schema.clone(),
        vec![
            Arc::new(index_arr),
            Arc::new(rows_list),
            Arc::new(unfiltered_list),
        ],
    )?)
}

/// Builder for final output rows that uses record-batch interleave to avoid scalar materialization.
struct OutputRowBuilder {
    sources: Vec<RecordBatch>,
    // Keep the underlying StructArray allocations alive so Arc pointers used as keys in
    // `source_map` cannot be re-used after drop (which would corrupt indices).
    source_ids: Vec<Arc<StructArray>>,
    source_map: HashMap<usize, usize>,
    indices: Vec<(usize, usize)>,
}

impl OutputRowBuilder {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            source_ids: Vec::new(),
            source_map: HashMap::new(),
            indices: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.sources.clear();
        self.source_ids.clear();
        self.source_map.clear();
        self.indices.clear();
    }

    fn clear_sources(&mut self) {
        self.sources.clear();
        self.source_ids.clear();
        self.source_map.clear();
    }

    fn compact_sources(&mut self) {
        if self.indices.is_empty() {
            self.clear_sources();
            return;
        }

        let used = used_source_mask(self.sources.len(), &self.indices);

        let mut remap = vec![usize::MAX; self.sources.len()];
        let mut new_sources = Vec::with_capacity(self.sources.len());
        let mut new_ids = Vec::with_capacity(self.source_ids.len());
        let mut new_map = HashMap::with_capacity(self.source_map.len());

        for i in 0..self.sources.len() {
            if !used[i] {
                continue;
            }
            let new_i = new_sources.len();
            new_sources.push(self.sources[i].clone());
            new_ids.push(self.source_ids[i].clone());
            remap[i] = new_i;
            let key = Arc::as_ptr(&new_ids[new_i]) as usize;
            new_map.insert(key, new_i);
        }

        for (src_i, _) in &mut self.indices {
            *src_i = remap[*src_i];
            debug_assert_ne!(*src_i, usize::MAX);
        }

        self.sources = new_sources;
        self.source_ids = new_ids;
        self.source_map = new_map;
    }

    fn register_source(&mut self, schema: &SchemaRef, source: &Arc<StructArray>) -> Result<usize> {
        let key = Arc::as_ptr(source) as usize;
        if let Some(existing) = self.source_map.get(&key) {
            return Ok(*existing);
        }

        // Important: `StructArray` can have a non-zero offset. `StructArray::column(i)` returns an
        // offset-aware child array, while `columns()` may expose the raw children.
        let mut cols = Vec::with_capacity(source.num_columns());
        let struct_len = source.len();
        for i in 0..source.num_columns() {
            let col = source.column(i).clone();
            debug_assert_eq!(col.len(), struct_len);
            cols.push(col);
        }
        let batch = RecordBatch::try_new(schema.clone(), cols)?;
        debug_assert_eq!(batch.num_rows(), source.len());
        let idx = self.sources.len();
        self.sources.push(batch);
        self.source_ids.push(source.clone());
        self.source_map.insert(key, idx);
        Ok(idx)
    }

    fn push_candidates(&mut self, schema: &SchemaRef, candidates: &[Candidate]) -> Result<()> {
        self.indices.reserve(candidates.len());
        for c in candidates {
            debug_assert!(c.row.row < c.row.data.len());
            let src_i = self.register_source(schema, &c.row.data)?;
            self.indices.push((src_i, c.row.row));
        }
        Ok(())
    }

    fn push_spill_batch(&mut self, schema: &SchemaRef, batch: &RecordBatch) -> Result<()> {
        // batch schema: [index, rows, unfiltered_dists]
        // rows is List<Struct<row, dist>>
        let rows_col = batch.column(1).as_list::<i32>();
        let values_struct = rows_col.values().as_struct(); // Struct<row, dist>
        let row_component = Arc::new(values_struct.column(0).as_struct().clone());
        let row_offsets = rows_col.value_offsets();

        // Register the row-component source once.
        let src_i = self.register_source(schema, &row_component)?;
        let value_len = row_component.len();

        for row in 0..batch.num_rows() {
            if rows_col.is_null(row) {
                continue;
            }
            let start = row_offsets[row] as usize;
            let end = row_offsets[row + 1] as usize;
            debug_assert!(end <= value_len);
            self.indices.reserve(end - start);
            for i in start..end {
                self.indices.push((src_i, i));
            }
        }
        Ok(())
    }

    fn take_ready(
        &mut self,
        schema: &SchemaRef,
        target_rows: usize,
    ) -> Result<Option<RecordBatch>> {
        if self.indices.len() < target_rows {
            return Ok(None);
        }
        self.take_any_up_to(schema, target_rows)
    }

    fn take_any_up_to(
        &mut self,
        schema: &SchemaRef,
        max_rows: usize,
    ) -> Result<Option<RecordBatch>> {
        if self.indices.is_empty() {
            return Ok(None);
        }
        let take = self.indices.len().min(max_rows.max(1));
        let taken: Vec<(usize, usize)> = self.indices.drain(..take).collect();

        let refs: Vec<&RecordBatch> = self.sources.iter().collect();

        let batch = interleave_record_batch(&refs, &taken)?;
        let batch = compact_batch(batch)?;
        debug_assert_eq!(batch.schema().as_ref(), schema.as_ref());

        // Drop any unreferenced sources immediately, so retained memory is bounded by the
        // remaining staged indices.
        self.compact_sources();

        Ok(Some(batch))
    }
}

struct PendingProbe {
    idx: Option<usize>,
    candidates: Vec<Candidate>,
    /// Unfiltered distances from previous spill for `idx` (top-K so far).
    prev_unfiltered: Vec<f64>,
    /// Unfiltered distances seen for `idx` from the current indexed partition.
    new_unfiltered: Vec<f64>,
}

impl PendingProbe {
    fn new() -> Self {
        Self {
            idx: None,
            candidates: Vec::new(),
            prev_unfiltered: Vec::new(),
            new_unfiltered: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.idx = None;
        self.candidates.clear();
        self.prev_unfiltered.clear();
        self.new_unfiltered.clear();
    }

    fn init_for_index(&mut self, idx: usize) {
        self.idx = Some(idx);
        self.candidates.clear();
        self.prev_unfiltered.clear();
        self.new_unfiltered.clear();
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    dist: f64,
    // Row data for this candidate.
    //
    // Performance note: we intentionally avoid materializing `ScalarValue` eagerly for every
    // joined row. Many joins produce far more than K matches per probe index, but we only need
    // the top-K (or ties) after sorting/pruning. We therefore keep a cheap reference to the
    // backing `StructArray` + row index, and use Arrow interleave to materialize output/spill
    // batches when needed.
    row: CandidateRowRef,
}

#[derive(Debug, Clone)]
struct CandidateRowRef {
    data: Arc<StructArray>,
    row: usize,
}

impl CandidateRowRef {
    // Intentionally no ScalarValue materialization here.
}

struct SpillRowReader {
    reader: RecordBatchSpillReader,
    /// Current batch loaded from file
    current_batch: Option<RecordBatch>,
    /// Current index within the batch
    current_offset: usize,
}

impl SpillRowReader {
    fn new(file: &RefCountedTempFile) -> Result<Self> {
        Ok(Self {
            reader: RecordBatchSpillReader::try_new(file)?,
            current_batch: None,
            current_offset: 0,
        })
    }

    /// Peeks the index of the next available row in the spill file.
    /// Returns None if EOF.
    fn peek_index(&mut self) -> Result<Option<usize>> {
        self.ensure_batch()?;
        if let Some(batch) = &self.current_batch {
            let index_col = batch
                .column(0)
                .as_primitive::<arrow::datatypes::UInt64Type>();
            Ok(Some(index_col.value(self.current_offset) as usize))
        } else {
            Ok(None)
        }
    }

    /// Reads the next row from spill file.
    /// Returns a batch slice containing consecutive rows with index < `until_idx`.
    /// Advances the internal cursor past the returned rows.
    fn take_prefix_before(&mut self, until_idx: usize) -> Result<Option<RecordBatch>> {
        self.ensure_batch()?;
        let Some(batch) = &self.current_batch else {
            return Ok(None);
        };

        let index_col = batch
            .column(0)
            .as_primitive::<arrow::datatypes::UInt64Type>();
        let values = index_col.values().as_ref();

        let start = self.current_offset;
        let slice = &values[start..batch.num_rows()];
        let rel_end = slice.partition_point(|v| (*v as usize) < until_idx);
        let end = start + rel_end;

        if end == start {
            return Ok(None);
        }

        let out = batch.slice(start, end - start);
        self.current_offset = end;
        Ok(Some(out))
    }

    fn ensure_batch(&mut self) -> Result<()> {
        if self.current_batch.is_none()
            || self.current_offset
                >= self
                    .current_batch
                    .as_ref()
                    .map(|b| b.num_rows())
                    .unwrap_or(0)
        {
            match self.reader.next_batch() {
                None => {
                    self.current_batch = None;
                }
                Some(batch_res) => {
                    self.current_batch = Some(batch_res?);
                    self.current_offset = 0;
                }
            }
        }
        Ok(())
    }
}

impl KNNResultsMerger {
    pub fn new(
        k: usize,
        include_tie_breaker: bool,
        target_batch_size: usize,
        runtime_env: Arc<RuntimeEnv>,
        spill_compression: SpillCompression,
        result_schema: SchemaRef,
        spill_metrics: SpillMetrics,
    ) -> Self {
        let spill_schema = Self::create_spill_schema(result_schema.clone());
        let target_spilled_batch_size = target_batch_size.div_ceil(k).max(1);
        Self {
            k,
            include_tie_breaker,
            target_batch_size,
            target_spilled_batch_size,
            result_schema,
            spill_schema,
            runtime_env,
            spill_compression,
            spill_metrics,
            state: Mutex::new(MergerState {
                previous_file: None,
                previous_reader: None,
                current_writer: None,
                pending: PendingProbe::new(),
                spill_builder: SpillRowBuilder::new(),
                spill_stage: SpillBatchStage::new(),
                output_builder: OutputRowBuilder::new(),
            }),
        }
    }

    pub fn is_single_partitioned(&self) -> bool {
        let state = self.state.lock();
        state.previous_file.is_none() && state.current_writer.is_none()
    }

    fn create_spill_schema(result_schema: SchemaRef) -> SchemaRef {
        // Schema:
        // index: UInt64
        // rows: List<Struct<row: Struct<...>, dist: Float64>>
        // unfiltered_dists: List<Float64> (top-K unfiltered distances so far)

        let index_field = Field::new("index", DataType::UInt64, false);

        let row_field = Field::new(
            "row",
            DataType::Struct(result_schema.fields().clone()),
            false,
        );
        let dist_field = Field::new("dist", DataType::Float64, false);

        let struct_fields = vec![row_field, dist_field];
        let struct_type = DataType::Struct(struct_fields.into());

        let rows_field = Field::new(
            "rows",
            DataType::List(Arc::new(Field::new("item", struct_type, true))),
            true,
        );
        let unfiltered_dists_field = Field::new(
            "unfiltered_dists",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        );

        Arc::new(Schema::new(vec![
            index_field,
            rows_field,
            unfiltered_dists_field,
        ]))
    }

    pub fn rotate(&self, probing_last_index: bool) -> Result<()> {
        let mut state = self.state.lock();

        // Flush any buffered probe row into spill builder.
        if state.current_writer.is_some() {
            self.flush_pending(&mut *state)?;

            // Preserve ordering: move any builder contents into staged spill batches.
            self.stage_all_spill_builder(&mut *state)?;

            // Drain remaining spill rows from previous reader.
            while let Some(batch) = match state.previous_reader.as_mut() {
                None => None,
                Some(reader) => reader.take_prefix_before(usize::MAX)?,
            } {
                let MergerState {
                    spill_stage,
                    current_writer,
                    ..
                } = &mut *state;
                spill_stage.push_batch(batch, self.target_spilled_batch_size);
                if let Some(writer) = current_writer.as_mut() {
                    spill_stage.maybe_flush(
                        writer,
                        &self.spill_schema,
                        self.target_spilled_batch_size,
                    )?;
                }
            }

            {
                let MergerState {
                    spill_stage,
                    current_writer,
                    ..
                } = &mut *state;
                if let Some(writer) = current_writer.as_mut() {
                    spill_stage.flush_all(
                        writer,
                        &self.spill_schema,
                        self.target_spilled_batch_size,
                    )?;
                }
            }
        }

        state.previous_file = state
            .current_writer
            .take()
            .map(|w| w.finish())
            .transpose()?;
        state.previous_reader = None;
        state.pending.reset();
        state.spill_builder.reset();
        state.spill_stage = SpillBatchStage::new();
        state.output_builder.reset();

        if let Some(file) = &state.previous_file {
            state.previous_reader = Some(SpillRowReader::new(file)?);
        }

        if !probing_last_index {
            state.current_writer = Some(RecordBatchSpillWriter::try_new(
                self.runtime_env.clone(),
                self.spill_schema.clone(),
                "knn_spill",
                self.spill_compression,
                self.spill_metrics.clone(),
                None,
            )?);
        }

        Ok(())
    }

    pub fn init_for_partition_0(&self, is_last: bool) -> Result<()> {
        let mut state = self.state.lock();
        if !is_last {
            state.current_writer = Some(RecordBatchSpillWriter::try_new(
                self.runtime_env.clone(),
                self.spill_schema.clone(),
                "knn_spill",
                self.spill_compression,
                self.spill_metrics.clone(),
                None,
            )?);
        }
        Ok(())
    }

    pub fn ingest(
        &self,
        joined_batch: RecordBatch,
        filtered_distances: Option<&[f64]>,
        filtered_probe_indices: &[u32],
        offset_in_partition: usize,
        unfiltered_distances: &[f64],
        unfiltered_probe_indices: &[u32],
    ) -> Result<Option<RecordBatch>> {
        if self.is_single_partitioned() {
            return Ok(Some(joined_batch));
        }

        let Some(filtered_distances) = filtered_distances else {
            return sedona_internal_err!("distances missing for KNN join");
        };

        if filtered_distances.len() != filtered_probe_indices.len() {
            return sedona_internal_err!(
                "filtered distances and probe indices length mismatch: {} vs {}",
                filtered_distances.len(),
                filtered_probe_indices.len()
            );
        }

        if unfiltered_distances.len() != unfiltered_probe_indices.len() {
            return sedona_internal_err!(
                "unfiltered distances and probe indices length mismatch: {} vs {}",
                unfiltered_distances.len(),
                unfiltered_probe_indices.len()
            );
        }

        let mut state = self.state.lock();
        let mut filtered_cursor = 0;
        let num_filtered = filtered_probe_indices.len();

        let mut unfiltered_cursor = 0;
        let num_unfiltered = unfiltered_probe_indices.len();

        // Consume `joined_batch` once we know we're in the multi-partition path.
        let joined_struct = Arc::new(StructArray::from(joined_batch));

        // The input indices are expected to be grouped by probe index (non-decreasing).
        // If this invariant changes upstream, the merge logic in this file would be incorrect.
        debug_assert!(
            unfiltered_probe_indices.windows(2).all(|w| w[0] <= w[1]),
            "unfiltered_probe_indices must be non-decreasing"
        );
        debug_assert!(
            filtered_probe_indices.windows(2).all(|w| w[0] <= w[1]),
            "filtered_probe_indices must be non-decreasing"
        );

        while unfiltered_cursor < num_unfiltered {
            // Probe indices are per-probe-batch. Use local probe idx for grouping, and only add
            // the partition offset when interacting with spill/global state.
            let local_probe_idx = unfiltered_probe_indices[unfiltered_cursor] as usize;
            let global_idx = offset_in_partition + local_probe_idx;

            if let Some(pending_idx) = state.pending.idx {
                debug_assert!(
                    global_idx >= pending_idx,
                    "probe indices must be non-decreasing within a partition"
                );
            }

            // If we moved past a buffered probe index, flush it now.
            if let Some(pending) = state.pending.idx {
                if global_idx != pending {
                    self.flush_pending(&mut state)?;
                }
            }

            self.gap_fill(&mut state, global_idx)?;

            let unfiltered_start = unfiltered_cursor;
            while unfiltered_cursor < num_unfiltered
                && unfiltered_probe_indices[unfiltered_cursor] as usize == local_probe_idx
            {
                unfiltered_cursor += 1;
            }
            let unfiltered_end = unfiltered_cursor;

            let filtered_start = filtered_cursor;
            while filtered_cursor < num_filtered
                && filtered_probe_indices[filtered_cursor] as usize == local_probe_idx
            {
                filtered_cursor += 1;
            }
            let filtered_end = filtered_cursor;

            // Initialize pending state for this probe index if needed, and merge from previous.
            if state.pending.idx != Some(global_idx) {
                self.init_pending_for_index(&mut state, global_idx)?;
            }

            // Collect unfiltered distances for this probe index from the current partition.
            state
                .pending
                .new_unfiltered
                .reserve(unfiltered_end - unfiltered_start);
            for i in unfiltered_start..unfiltered_end {
                state.pending.new_unfiltered.push(unfiltered_distances[i]);
            }

            // Collect new (filtered) candidates for this probe index from the current joined batch.
            state
                .pending
                .candidates
                .reserve(filtered_end.saturating_sub(filtered_start));
            for i in filtered_start..filtered_end {
                state.pending.candidates.push(Candidate {
                    dist: filtered_distances[i],
                    row: CandidateRowRef {
                        data: joined_struct.clone(),
                        row: i,
                    },
                });
            }
        }

        // Do not flush the last buffered probe index here. The final probe index in a produced
        // slice can be split across multiple `ingest` calls due to `max_batch_size`.
        // We instead flush it when we observe the next probe index, or when the probe batch ends
        // via `produce_last_batch()`.

        // If we are in the final partition (no spill writer), emit output in ~target_batch_size chunks.
        if state.current_writer.is_none() {
            state
                .output_builder
                .take_ready(&self.result_schema, self.target_batch_size)
        } else {
            Ok(None)
        }
    }

    /// Flushes any pending buffered probe index at the end of a probe batch iterator.
    ///
    /// This is used to emit the final probe index that may have been kept buffered because
    /// it could continue in the next produced slice.
    ///
    /// Returns `Ok(Some(batch))` at most once per pending buffered index; if there is nothing
    /// pending (or results are being spilled to disk for non-final indexed partitions), returns
    /// `Ok(None)`.
    pub fn produce_batch_until(
        &self,
        end_global_idx_exclusive: usize,
    ) -> Result<Option<RecordBatch>> {
        if self.is_single_partitioned() {
            return Ok(None);
        }

        let mut state = self.state.lock();

        // If we already have buffered output ready, emit it first.
        if state.current_writer.is_none() {
            if let Some(batch) = state
                .output_builder
                .take_ready(&self.result_schema, self.target_batch_size)?
            {
                return Ok(Some(batch));
            }
        }

        // Only flush the currently pending index. We must not drain spill rows beyond the
        // caller-provided probe-row range end, since future probe batches (with larger global
        // indices) may still need to merge against those rows.
        self.flush_pending(&mut state)?;

        if state.current_writer.is_none() {
            // Drain any remaining spill rows that belong to the current probe-row range.
            // This is required when the last indexed partition has few/no matches for some
            // probe rows: those results must still be emitted from the previous spill file.
            self.gap_fill(&mut state, end_global_idx_exclusive)?;
            // Final flush: allow returning a smaller tail batch.
            state
                .output_builder
                .take_any_up_to(&self.result_schema, self.target_batch_size)
        } else {
            Ok(None)
        }
    }

    fn flush_pending(&self, state: &mut MergerState) -> Result<()> {
        let Some(idx) = state.pending.idx else {
            return Ok(());
        };

        // Work on the buffered vectors in-place to preserve capacity.
        let pending = &mut state.pending;
        let candidates = &mut pending.candidates;
        let prev_unfiltered = &pending.prev_unfiltered;
        let new_unfiltered = &pending.new_unfiltered;

        // Sort by distance.
        candidates.sort_by(|a, b| a.dist.total_cmp(&b.dist));

        let merged_unfiltered = self.merge_unfiltered_topk(&prev_unfiltered, &new_unfiltered);
        let threshold = if merged_unfiltered.len() >= self.k {
            Some(merged_unfiltered[self.k - 1])
        } else {
            None
        };

        // Select candidates according to threshold + tie-breaker behavior.
        // This returns a prefix length into the sorted `candidates` slice.
        let selected_len = self.selected_prefix_len(candidates, threshold);
        let selected = &candidates[..selected_len];

        if state.current_writer.is_some() {
            let MergerState {
                current_writer,
                spill_builder,
                spill_stage,
                ..
            } = state;
            let writer = current_writer
                .as_mut()
                .expect("current_writer must be present in spill mode");

            spill_builder.push_row(idx, selected, &merged_unfiltered)?;

            // When enough probe rows are buffered, materialize a spill batch and stage/flush it.
            while spill_builder.len_rows() >= self.target_spilled_batch_size {
                if let Some(batch) = spill_builder.take_batch(
                    &self.spill_schema,
                    &self.result_schema,
                    self.target_spilled_batch_size,
                )? {
                    spill_stage.push_batch(batch, self.target_spilled_batch_size);
                    spill_stage.maybe_flush(
                        writer,
                        &self.spill_schema,
                        self.target_spilled_batch_size,
                    )?;
                }
            }
        } else {
            // Final partition: buffer flat output rows and let caller pull in target-sized chunks.
            state
                .output_builder
                .push_candidates(&self.result_schema, selected)?;
        }

        pending.reset();
        Ok(())
    }

    fn merge_unfiltered_topk(&self, prev: &[f64], new: &[f64]) -> Vec<f64> {
        let mut all = Vec::with_capacity(prev.len() + new.len());
        all.extend_from_slice(prev);
        all.extend_from_slice(new);

        // Keep only the K smallest distances, sorted.
        // This avoids a full sort when prev+new is large.
        if self.k > 0 && all.len() > self.k {
            let kth = self.k - 1;
            all.select_nth_unstable_by(kth, |a, b| a.total_cmp(b));
            all.truncate(self.k);
        }
        all.sort_by(|a, b| a.total_cmp(b));
        all
    }

    fn selected_prefix_len(&self, candidates: &[Candidate], threshold: Option<f64>) -> usize {
        if candidates.is_empty() {
            return 0;
        }

        // Candidates are already sorted by distance.
        let mut end = match threshold {
            Some(t) => {
                // Include candidates with dist <= threshold.
                candidates.partition_point(|c| c.dist <= t)
            }
            None => candidates.len(),
        };

        if !self.include_tie_breaker {
            end = end.min(self.k);
        }
        end
    }

    fn gap_fill(&self, state: &mut MergerState, until_idx: usize) -> Result<()> {
        if state.previous_reader.is_none() {
            return Ok(());
        };

        loop {
            let peek_opt = {
                let reader = state
                    .previous_reader
                    .as_mut()
                    .expect("previous_reader must be present");
                reader.peek_index()?
            };
            let peek = match peek_opt {
                Some(v) => v,
                None => break,
            };
            if peek >= until_idx {
                break;
            }

            // Pull as many rows as possible in a single batch slice.
            let batch_opt = {
                let reader = state
                    .previous_reader
                    .as_mut()
                    .expect("previous_reader must be present");
                reader.take_prefix_before(until_idx)?
            };
            let batch = match batch_opt {
                Some(b) => b,
                None => break,
            };

            if state.current_writer.is_some() {
                // Preserve ordering: stage any builder contents before staging gap-fill carry-over.
                self.stage_all_spill_builder(state)?;

                let MergerState {
                    spill_stage,
                    current_writer,
                    ..
                } = state;
                spill_stage.push_batch(batch, self.target_spilled_batch_size);
                if let Some(writer) = current_writer.as_mut() {
                    spill_stage.maybe_flush(
                        writer,
                        &self.spill_schema,
                        self.target_spilled_batch_size,
                    )?;
                }
            } else {
                // Final partition: flatten spill rows into output builder using interleave.
                state
                    .output_builder
                    .push_spill_batch(&self.result_schema, &batch)?;
            }
        }
        Ok(())
    }

    fn stage_all_spill_builder(&self, state: &mut MergerState) -> Result<()> {
        if state.spill_builder.is_empty() {
            return Ok(());
        }
        // Materialize whatever is currently buffered so that later staged batches preserve order.
        if let Some(batch) =
            state
                .spill_builder
                .take_batch(&self.spill_schema, &self.result_schema, usize::MAX)?
        {
            state
                .spill_stage
                .push_batch(batch, self.target_spilled_batch_size);
        }
        Ok(())
    }

    fn init_pending_for_index(&self, state: &mut MergerState, global_idx: usize) -> Result<()> {
        state.pending.init_for_index(global_idx);

        self.merge_from_previous_for_index(state, global_idx)
    }

    fn merge_from_previous_for_index(
        &self,
        state: &mut MergerState,
        global_idx: usize,
    ) -> Result<()> {
        // Drain all spill rows for this global index, working in-batch (no 1-row slicing).
        loop {
            let Some(reader) = state.previous_reader.as_mut() else {
                break;
            };

            reader.ensure_batch()?;
            let Some(batch) = reader.current_batch.as_ref() else {
                break;
            };

            let index_col = batch
                .column(0)
                .as_primitive::<arrow::datatypes::UInt64Type>();
            let values = index_col.values().as_ref();

            if reader.current_offset >= batch.num_rows() {
                // ensure_batch() should have advanced, but be defensive.
                break;
            }

            let current = values[reader.current_offset] as usize;
            if current != global_idx {
                break;
            }

            let start = reader.current_offset;
            let slice = &values[start..batch.num_rows()];
            let rel_end = slice.partition_point(|v| (*v as usize) == global_idx);
            let end = start + rel_end;
            reader.current_offset = end;

            self.extract_from_spill_range(
                batch,
                start..end,
                &mut state.pending.candidates,
                &mut state.pending.prev_unfiltered,
            )?;
        }

        Ok(())
    }

    fn extract_from_spill_range(
        &self,
        batch: &RecordBatch,
        row_range: Range<usize>,
        candidates: &mut Vec<Candidate>,
        unfiltered_dists: &mut Vec<f64>,
    ) -> Result<()> {
        // batch schema: [index, rows, unfiltered_dists]
        // rows is List<Struct<row, dist>>
        let rows_col = batch.column(1).as_list::<i32>();
        let values_struct = rows_col.values().as_struct(); // Struct<row, dist>
        let row_component = values_struct.column(0).as_struct();
        let dist_component = values_struct
            .column(1)
            .as_primitive::<arrow::datatypes::Float64Type>();

        // Avoid allocating a fresh Arc per candidate; the struct array itself is cheap to
        // clone (Arc-backed), so we wrap it once and clone the Arc.
        let row_component = Arc::new(row_component.clone());
        let row_offsets = rows_col.value_offsets();

        for row in row_range.clone() {
            if rows_col.is_null(row) {
                continue;
            }
            let start = row_offsets[row] as usize;
            let end = row_offsets[row + 1] as usize;
            for i in start..end {
                let d = dist_component.value(i);
                candidates.push(Candidate {
                    dist: d,
                    row: CandidateRowRef {
                        data: row_component.clone(),
                        row: i,
                    },
                });
            }
        }

        // unfiltered_dists: List<Float64>
        let unfiltered_col = batch.column(2).as_list::<i32>();
        let unfiltered_offsets = unfiltered_col.value_offsets();
        let unfiltered_values = unfiltered_col
            .values()
            .as_primitive::<arrow::datatypes::Float64Type>();

        for row in row_range {
            if unfiltered_col.is_null(row) {
                continue;
            }
            let start = unfiltered_offsets[row] as usize;
            let end = unfiltered_offsets[row + 1] as usize;
            for i in start..end {
                unfiltered_dists.push(unfiltered_values.value(i));
            }
        }

        Ok(())
    }

    // Note: build_spill_batch/build_result_batch/flatten_spill_batch were intentionally removed.
    // The new implementation buffers rows and uses Arrow interleave to materialize batches,
    // avoiding scalar materialization and concat of tiny RecordBatches.
}
