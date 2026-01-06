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

use std::sync::Arc;

use arrow::array::{
    Array, AsArray, Float64Builder, ListArray, RecordBatch, StructArray, UInt64Builder,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::config::SpillCompression;
use datafusion_common::{Result, ScalarValue};
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_plan::metrics::SpillMetrics;
use parking_lot::Mutex;
use sedona_common::sedona_internal_err;

use crate::utils::spill::{RecordBatchSpillReader, RecordBatchSpillWriter};

/// KNNResultsMerger handles the merging of KNN "nearest so far" results from multiple partitions.
/// It maintains spill files to store intermediate results.
pub struct KNNResultsMerger {
    k: usize,
    include_tie_breaker: bool,
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
    pending_idx: Option<usize>,
    pending_candidates: Vec<Candidate>,
    /// Unfiltered distances from previous spill for `pending_idx` (top-K so far).
    pending_prev_unfiltered: Vec<f64>,
    /// Unfiltered distances seen for `pending_idx` from the current indexed partition.
    pending_new_unfiltered: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Candidate {
    dist: f64,
    // Row data for this candidate.
    //
    // Performance note: we intentionally avoid materializing `ScalarValue` eagerly for every
    // joined row. Many joins produce far more than K matches per probe index, but we only need
    // the top-K (or ties) after sorting/pruning. We therefore keep a cheap reference to the
    // backing `StructArray` + row index, and only convert to `ScalarValue` for the selected
    // candidates when building output/spill batches.
    row: CandidateRowRef,
}

#[derive(Debug, Clone)]
struct CandidateRowRef {
    data: Arc<StructArray>,
    row: usize,
}

impl CandidateRowRef {
    fn to_scalar(&self) -> Result<ScalarValue> {
        ScalarValue::try_from_array(self.data.as_ref(), self.row)
    }
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
    /// Returns (index, row_batch) where row_batch is a slice of size 1.
    fn next_row(&mut self) -> Result<Option<(usize, RecordBatch)>> {
        self.ensure_batch()?;
        if let Some(batch) = &self.current_batch {
            let index_col = batch
                .column(0)
                .as_primitive::<arrow::datatypes::UInt64Type>();
            let idx = index_col.value(self.current_offset) as usize;

            // Slice the batch for 1 row
            let row_batch = batch.slice(self.current_offset, 1);

            self.current_offset += 1;
            Ok(Some((idx, row_batch)))
        } else {
            Ok(None)
        }
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
        runtime_env: Arc<RuntimeEnv>,
        spill_compression: SpillCompression,
        result_schema: SchemaRef,
        spill_metrics: SpillMetrics,
    ) -> Self {
        let spill_schema = Self::create_spill_schema(result_schema.clone());
        Self {
            k,
            include_tie_breaker,
            result_schema,
            spill_schema,
            runtime_env,
            spill_compression,
            spill_metrics,
            state: Mutex::new(MergerState {
                previous_file: None,
                previous_reader: None,
                current_writer: None,
                pending_idx: None,
                pending_candidates: Vec::new(),
                pending_prev_unfiltered: Vec::new(),
                pending_new_unfiltered: Vec::new(),
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

        // Flush any buffered probe row into the current spill before rotating.
        // Note: rotate is only called when there *is* a next indexed partition.
        if state.current_writer.is_some() {
            let mut ignored_output: Vec<RecordBatch> = Vec::new();
            self.flush_pending(&mut *state, &mut ignored_output)?;
        }

        // Drain any remaining rows from previous reader.
        while let Some(reader) = state.previous_reader.as_mut() {
            let Some(_) = reader.peek_index()? else {
                break;
            };
            if let Some((_, row_batch)) = reader.next_row()? {
                if let Some(writer) = &mut state.current_writer {
                    writer.write_batch(&row_batch)?;
                }
            }
        }

        state.previous_file = state
            .current_writer
            .take()
            .map(|w| w.finish())
            .transpose()?;
        state.previous_reader = None;
        state.pending_idx = None;
        state.pending_candidates.clear();
        state.pending_prev_unfiltered.clear();
        state.pending_new_unfiltered.clear();

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
        let mut output_batches: Vec<RecordBatch> = Vec::new();

        let mut filtered_cursor = 0;
        let num_filtered = filtered_probe_indices.len();

        let mut unfiltered_cursor = 0;
        let num_unfiltered = unfiltered_probe_indices.len();

        // Consume `joined_batch` once we know we're in the multi-partition path.
        let joined_struct = Arc::new(StructArray::from(joined_batch));

        while unfiltered_cursor < num_unfiltered {
            // Probe indices are per-probe-batch. Use local probe idx for grouping, and only add
            // the partition offset when interacting with spill/global state.
            let local_probe_idx = unfiltered_probe_indices[unfiltered_cursor] as usize;
            let global_idx = offset_in_partition + local_probe_idx;

            // If we moved past a buffered probe index, flush it now.
            if let Some(pending) = state.pending_idx {
                if global_idx != pending {
                    self.flush_pending(&mut state, &mut output_batches)?;
                }
            }

            self.gap_fill(&mut state, global_idx, &mut output_batches)?;

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
            if state.pending_idx != Some(global_idx) {
                self.init_pending_for_index(&mut state, global_idx)?;
            }

            // Collect unfiltered distances for this probe index from the current partition.
            state
                .pending_new_unfiltered
                .reserve(unfiltered_end - unfiltered_start);
            for i in unfiltered_start..unfiltered_end {
                state.pending_new_unfiltered.push(unfiltered_distances[i]);
            }

            // Collect new (filtered) candidates for this probe index from the current joined batch.
            state
                .pending_candidates
                .reserve(filtered_end.saturating_sub(filtered_start));
            for i in filtered_start..filtered_end {
                state.pending_candidates.push(Candidate {
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

        Self::concat_output_batches(output_batches)
    }

    /// Flushes any pending buffered probe index at the end of a probe batch iterator.
    ///
    /// This is used to emit the final probe index that may have been kept buffered because
    /// it could continue in the next produced slice.
    ///
    /// Returns `Ok(Some(batch))` at most once per pending buffered index; if there is nothing
    /// pending (or results are being spilled to disk for non-final indexed partitions), returns
    /// `Ok(None)`.
    pub fn produce_last_batch(&self) -> Result<Option<RecordBatch>> {
        if self.is_single_partitioned() {
            return Ok(None);
        }

        let mut state = self.state.lock();
        let mut output_batches: Vec<RecordBatch> = Vec::new();

        // Only flush the currently pending index; do not drain remaining spill rows here.
        // Draining would be incorrect because future probe batches (with larger global indices)
        // may still need to merge against those rows.
        self.flush_pending(&mut state, &mut output_batches)?;

        Self::concat_output_batches(output_batches)
    }

    fn concat_output_batches(output_batches: Vec<RecordBatch>) -> Result<Option<RecordBatch>> {
        if output_batches.is_empty() {
            return Ok(None);
        }
        let schema = output_batches[0].schema();
        let batch = arrow::compute::concat_batches(&schema, &output_batches)?;
        Ok(Some(batch))
    }

    fn flush_pending(&self, state: &mut MergerState, output: &mut Vec<RecordBatch>) -> Result<()> {
        let Some(idx) = state.pending_idx else {
            return Ok(());
        };

        let mut candidates = std::mem::take(&mut state.pending_candidates);
        let prev_unfiltered = std::mem::take(&mut state.pending_prev_unfiltered);
        let new_unfiltered = std::mem::take(&mut state.pending_new_unfiltered);
        state.pending_idx = None;

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
        let selected_len = self.selected_prefix_len(&candidates, threshold);
        let selected = &candidates[..selected_len];

        if let Some(writer) = &mut state.current_writer {
            let batch = self.build_spill_batch(idx, selected, &merged_unfiltered)?;
            writer.write_batch(&batch)?;
        } else {
            if let Some(batch) = self.build_result_batch(selected)? {
                output.push(batch);
            }
        }
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

    fn gap_fill(
        &self,
        state: &mut MergerState,
        until_idx: usize,
        output: &mut Vec<RecordBatch>,
    ) -> Result<()> {
        let Some(reader) = state.previous_reader.as_mut() else {
            return Ok(());
        };

        loop {
            let Some(peek) = reader.peek_index()? else {
                break;
            };
            if peek >= until_idx {
                break;
            }

            let (_, row_batch) = reader.next_row()?.unwrap();
            if let Some(writer) = &mut state.current_writer {
                writer.write_batch(&row_batch)?;
            } else {
                let flat = self.flatten_spill_batch(&row_batch)?;
                if flat.num_rows() > 0 {
                    output.push(flat);
                }
            }
        }
        Ok(())
    }

    fn init_pending_for_index(&self, state: &mut MergerState, global_idx: usize) -> Result<()> {
        state.pending_idx = Some(global_idx);
        state.pending_candidates.clear();
        state.pending_prev_unfiltered.clear();
        state.pending_new_unfiltered.clear();

        self.merge_from_previous_for_index(state, global_idx)
    }

    fn merge_from_previous_for_index(
        &self,
        state: &mut MergerState,
        global_idx: usize,
    ) -> Result<()> {
        // Drain all spill rows for this global index.
        loop {
            // Avoid holding a mutable borrow of `previous_reader` across the subsequent
            // mutations of other `state` fields.
            let row_batch = match state.previous_reader.as_mut() {
                None => break,
                Some(reader) => {
                    if reader.peek_index()? != Some(global_idx) {
                        break;
                    }
                    let (_, row_batch) = reader.next_row()?.unwrap();
                    row_batch
                }
            };

            self.extract_from_spill(
                &row_batch,
                &mut state.pending_candidates,
                &mut state.pending_prev_unfiltered,
            )?;
        }

        Ok(())
    }

    fn extract_from_spill(
        &self,
        batch: &RecordBatch,
        candidates: &mut Vec<Candidate>,
        unfiltered_dists: &mut Vec<f64>,
    ) -> Result<()> {
        // batch schema: [index, rows, unfiltered_dists]
        // rows is List<Struct<row, dist>>
        let rows_col = batch.column(1).as_list::<i32>();
        if rows_col.is_null(0) {
            // Still try to read unfiltered_dists even if rows are empty.
        } else {
            let values_struct = rows_col.values().as_struct(); // Struct<row, dist>
            let row_component = values_struct.column(0).as_struct();
            let dist_component = values_struct
                .column(1)
                .as_primitive::<arrow::datatypes::Float64Type>();

            // Avoid allocating a fresh Arc per candidate; the struct array itself is cheap to
            // clone (Arc-backed), so we wrap it once and clone the Arc.
            let row_component = Arc::new(row_component.clone());

            let start = rows_col.value_offsets()[0] as usize;
            let end = rows_col.value_offsets()[1] as usize;

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
        if !unfiltered_col.is_null(0) {
            let values = unfiltered_col
                .values()
                .as_primitive::<arrow::datatypes::Float64Type>();
            let start = unfiltered_col.value_offsets()[0] as usize;
            let end = unfiltered_col.value_offsets()[1] as usize;
            for i in start..end {
                unfiltered_dists.push(values.value(i));
            }
        }
        Ok(())
    }

    fn build_spill_batch(
        &self,
        idx: usize,
        candidates: &[Candidate],
        unfiltered_dists: &[f64],
    ) -> Result<RecordBatch> {
        let mut idx_builder = UInt64Builder::new();
        idx_builder.append_value(idx as u64);

        // Build rows from selected candidates.
        // We only materialize scalars for the selected set (<= K or ties).
        // inner struct: Struct<row, dist>
        let row_fields = self.result_schema.fields().clone();
        let combined_fields = vec![
            Field::new("row", DataType::Struct(row_fields.clone()), false),
            Field::new("dist", DataType::Float64, false),
        ];

        let mut dist_builder = Float64Builder::with_capacity(candidates.len());
        let mut row_scalars = Vec::with_capacity(candidates.len());

        for c in candidates {
            dist_builder.append_value(c.dist);
            row_scalars.push(c.row.to_scalar()?);
        }

        let row_array = if row_scalars.is_empty() {
            arrow::array::new_empty_array(&DataType::Struct(row_fields.into()))
        } else {
            ScalarValue::iter_to_array(row_scalars.into_iter())?
        };

        let combined_struct = StructArray::try_new(
            combined_fields.into(),
            vec![Arc::new(row_array), Arc::new(dist_builder.finish())],
            None,
        )?;

        // Use ListArray::try_new instead of builder
        let offsets = OffsetBuffer::<i32>::from_lengths(std::iter::once(combined_struct.len()));
        let list_field = Arc::new(Field::new(
            "item",
            combined_struct.data_type().clone(),
            true,
        ));
        let list_array = ListArray::try_new(list_field, offsets, Arc::new(combined_struct), None)?;

        // Build unfiltered_dists list
        let mut unfiltered_values = Float64Builder::with_capacity(unfiltered_dists.len());
        for d in unfiltered_dists {
            unfiltered_values.append_value(*d);
        }
        let unfiltered_values = unfiltered_values.finish();

        let unfiltered_offsets =
            OffsetBuffer::<i32>::from_lengths(std::iter::once(unfiltered_values.len()));
        let unfiltered_field = Arc::new(Field::new("item", DataType::Float64, true));
        let unfiltered_list = ListArray::try_new(
            unfiltered_field,
            unfiltered_offsets,
            Arc::new(unfiltered_values),
            None,
        )?;

        Ok(RecordBatch::try_new(
            self.spill_schema.clone(),
            vec![
                Arc::new(idx_builder.finish()),
                Arc::new(list_array),
                Arc::new(unfiltered_list),
            ],
        )?)
    }

    fn build_result_batch(&self, candidates: &[Candidate]) -> Result<Option<RecordBatch>> {
        if candidates.is_empty() {
            return Ok(None);
        }
        let mut scalars = Vec::with_capacity(candidates.len());
        for c in candidates {
            scalars.push(c.row.to_scalar()?);
        }
        let array = ScalarValue::iter_to_array(scalars.into_iter())?;
        let struct_arr = array.as_struct();
        Ok(Some(RecordBatch::try_new(
            self.result_schema.clone(),
            struct_arr.columns().to_vec(),
        )?))
    }

    fn flatten_spill_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let mut candidates = Vec::new();
        let mut unfiltered = Vec::new();
        self.extract_from_spill(batch, &mut candidates, &mut unfiltered)?;
        self.build_result_batch(&candidates)
            .map(|opt| opt.unwrap_or_else(|| RecordBatch::new_empty(self.result_schema.clone())))
    }
}
