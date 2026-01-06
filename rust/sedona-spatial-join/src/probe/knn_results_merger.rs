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

use std::cmp::Ordering;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use arrow::array::{
    Array, AsArray, Float64Builder, ListArray, RecordBatch, StructArray, UInt64Builder,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use datafusion::config::SpillCompression;
use datafusion_common::{Result, ScalarValue};
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_plan::metrics::SpillMetrics;
use parking_lot::Mutex;
use sedona_common::sedona_internal_err;

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
    /// File to write results for current (0..N) partitions
    current_file: Option<RefCountedTempFile>,

    /// Reader for previous file
    previous_reader: Option<SpillReader>,
    /// Writer for current file
    current_writer: Option<StreamWriter<File>>,

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
    data: ScalarValue, // Struct including row data
}

struct SpillReader {
    reader: StreamReader<BufReader<File>>,
    /// Current batch loaded from file
    current_batch: Option<RecordBatch>,
    /// Current index within the batch
    current_offset: usize,
}

impl SpillReader {
    fn new(file: &RefCountedTempFile) -> Result<Self> {
        let f = File::open(file.path())?;
        let reader = StreamReader::try_new(BufReader::new(f), None)?;
        Ok(Self {
            reader,
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
            if let Some(batch_res) = self.reader.next() {
                self.current_batch = Some(batch_res?);
                self.current_offset = 0;
            } else {
                self.current_batch = None;
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
                current_file: None,
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
        state.previous_file.is_none() && state.current_file.is_none()
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
            self.flush_pending(&mut *state, false, &mut Vec::new())?;
        }

        // Drain any remaining rows from previous reader
        while let Some(_) = state
            .previous_reader
            .as_mut()
            .and_then(|r| r.peek_index().ok().flatten())
        {
            if let Some((_, row_batch)) = state.previous_reader.as_mut().unwrap().next_row()? {
                if let Some(writer) = &mut state.current_writer {
                    writer.write(&row_batch)?;
                }
            }
        }

        if let Some(mut writer) = state.current_writer.take() {
            writer.finish()?;
        }

        state.previous_file = state.current_file.take();
        state.previous_reader = None;
        state.pending_idx = None;
        state.pending_candidates.clear();
        state.pending_prev_unfiltered.clear();
        state.pending_new_unfiltered.clear();

        if let Some(file) = &state.previous_file {
            state.previous_reader = Some(SpillReader::new(file)?);
        }

        if !probing_last_index {
            let file = self.runtime_env.disk_manager.create_tmp_file("knn_spill")?;
            let f = File::create(file.path())?;

            let mut opts = IpcWriteOptions::default();
            opts = opts.try_with_compression(self.spill_compression.into())?;

            let writer = StreamWriter::try_new_with_options(f, &self.spill_schema, opts)?;
            state.current_file = Some(file);
            state.current_writer = Some(writer);
            self.spill_metrics.spill_file_count.add(1);
        }

        Ok(())
    }

    pub fn init_for_partition_0(&self, is_last: bool) -> Result<()> {
        let mut state = self.state.lock();
        if !is_last {
            let file = self.runtime_env.disk_manager.create_tmp_file("knn_spill")?;
            let f = File::create(file.path())?;

            let mut opts = IpcWriteOptions::default();
            opts = opts.try_with_compression(self.spill_compression.into())?;

            let writer = StreamWriter::try_new_with_options(f, &self.spill_schema, opts)?;
            state.current_file = Some(file);
            state.current_writer = Some(writer);
            self.spill_metrics.spill_file_count.add(1);
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
        let mut output_batches = Vec::new();

        let mut filtered_cursor = 0;
        let num_filtered = filtered_probe_indices.len();

        let mut unfiltered_cursor = 0;
        let num_unfiltered = unfiltered_probe_indices.len();

        let joined_struct = StructArray::from(joined_batch.clone());

        while unfiltered_cursor < num_unfiltered {
            let batch_probe_idx = unfiltered_probe_indices[unfiltered_cursor] as usize;
            let global_idx = offset_in_partition + batch_probe_idx;

            // If we moved past a buffered probe index, flush it now.
            if let Some(pending) = state.pending_idx {
                if global_idx != pending {
                    self.flush_pending(&mut state, false, &mut output_batches)?;
                }
            }

            self.gap_fill(&mut state, global_idx, &mut output_batches)?;

            let unfiltered_start = unfiltered_cursor;
            while unfiltered_cursor < num_unfiltered
                && (unfiltered_probe_indices[unfiltered_cursor] as usize + offset_in_partition)
                    == global_idx
            {
                unfiltered_cursor += 1;
            }
            let unfiltered_end = unfiltered_cursor;

            let filtered_start = filtered_cursor;
            while filtered_cursor < num_filtered
                && (filtered_probe_indices[filtered_cursor] as usize + offset_in_partition)
                    == global_idx
            {
                filtered_cursor += 1;
            }
            let filtered_end = filtered_cursor;

            // Initialize pending state for this probe index if needed, and merge from previous.
            if state.pending_idx != Some(global_idx) {
                state.pending_idx = Some(global_idx);
                state.pending_candidates.clear();
                state.pending_prev_unfiltered.clear();
                state.pending_new_unfiltered.clear();

                loop {
                    let peek = match state.previous_reader.as_mut() {
                        Some(r) => r.peek_index().ok().flatten(),
                        None => None,
                    };

                    if peek == Some(global_idx) {
                        let (_, row_batch) = {
                            // Avoid holding a mutable borrow of `previous_reader` across the
                            // subsequent mutations of other `state` fields.
                            state.previous_reader.as_mut().unwrap().next_row()?.unwrap()
                        };

                        let mut pending_prev_unfiltered =
                            std::mem::take(&mut state.pending_prev_unfiltered);
                        self.extract_from_spill(
                            &row_batch,
                            &mut state.pending_candidates,
                            &mut pending_prev_unfiltered,
                        )?;
                        state.pending_prev_unfiltered = pending_prev_unfiltered;
                    } else {
                        break;
                    }
                }
            }

            // Collect unfiltered distances for this probe index from the current partition.
            for i in unfiltered_start..unfiltered_end {
                state.pending_new_unfiltered.push(unfiltered_distances[i]);
            }

            // Collect new (filtered) candidates for this probe index from the current joined batch.
            for i in filtered_start..filtered_end {
                let scalar = ScalarValue::try_from_array(&joined_struct, i)?;
                state.pending_candidates.push(Candidate {
                    dist: filtered_distances[i],
                    data: scalar,
                });
            }
        }

        // Do not flush the last buffered probe index here. The final probe index in a produced
        // slice can be split across multiple `ingest` calls due to `max_batch_size`.
        // We instead flush it when we observe the next probe index, or when the probe batch ends
        // via `produce_last_batch()`.

        if output_batches.is_empty() {
            Ok(None)
        } else {
            let schema = output_batches[0].schema();
            let batch = arrow::compute::concat_batches(&schema, &output_batches)?;
            Ok(Some(batch))
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
    pub fn produce_last_batch(&self) -> Result<Option<RecordBatch>> {
        if self.is_single_partitioned() {
            return Ok(None);
        }

        let mut state = self.state.lock();
        let mut output_batches = Vec::new();

        // Only flush the currently pending index; do not drain remaining spill rows here.
        // Draining would be incorrect because future probe batches (with larger global indices)
        // may still need to merge against those rows.
        self.flush_pending(&mut state, false, &mut output_batches)?;

        if output_batches.is_empty() {
            Ok(None)
        } else {
            let schema = output_batches[0].schema();
            let batch = arrow::compute::concat_batches(&schema, &output_batches)?;
            Ok(Some(batch))
        }
    }

    fn flush_pending(
        &self,
        state: &mut MergerState,
        require_pending: bool,
        output: &mut Vec<RecordBatch>,
    ) -> Result<()> {
        let Some(idx) = state.pending_idx else {
            return Ok(());
        };

        let mut candidates = std::mem::take(&mut state.pending_candidates);
        let prev_unfiltered = std::mem::take(&mut state.pending_prev_unfiltered);
        let new_unfiltered = std::mem::take(&mut state.pending_new_unfiltered);
        state.pending_idx = None;

        // Sort by distance.
        candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

        let merged_unfiltered = self.merge_unfiltered_topk(&prev_unfiltered, &new_unfiltered);
        let threshold = if merged_unfiltered.len() >= self.k {
            Some(merged_unfiltered[self.k - 1])
        } else {
            None
        };

        // Select candidates according to threshold + tie-breaker behavior.
        let selected = self.select_candidates(&candidates, threshold);

        if let Some(writer) = &mut state.current_writer {
            let batch = self.build_spill_batch(idx, &selected, &merged_unfiltered)?;
            writer.write(&batch)?;
        } else {
            if let Some(batch) = self.build_result_batch(&selected)? {
                output.push(batch);
            }
        }

        // require_pending is only used to make intent explicit at call sites; currently no-op.
        let _ = require_pending;
        Ok(())
    }

    fn merge_unfiltered_topk(&self, prev: &[f64], new: &[f64]) -> Vec<f64> {
        let mut all = Vec::with_capacity(prev.len() + new.len());
        all.extend_from_slice(prev);
        all.extend_from_slice(new);
        all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        if all.len() > self.k {
            all.truncate(self.k);
        }
        all
    }

    fn select_candidates(
        &self,
        candidates: &[Candidate],
        threshold: Option<f64>,
    ) -> Vec<Candidate> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let filtered: Vec<Candidate> = match threshold {
            Some(t) => candidates.iter().filter(|c| c.dist <= t).cloned().collect(),
            None => candidates.to_vec(),
        };

        if !self.include_tie_breaker {
            return filtered.into_iter().take(self.k).collect();
        }

        filtered
    }

    fn gap_fill(
        &self,
        state: &mut MergerState,
        until_idx: usize,
        output: &mut Vec<RecordBatch>,
    ) -> Result<()> {
        while let Some(peek) = state
            .previous_reader
            .as_mut()
            .and_then(|r| r.peek_index().ok().flatten())
        {
            if peek < until_idx {
                let (_, row_batch) = state.previous_reader.as_mut().unwrap().next_row()?.unwrap();
                if state.current_writer.is_some() {
                    state.current_writer.as_mut().unwrap().write(&row_batch)?;
                } else {
                    let flat = self.flatten_spill_batch(&row_batch)?;
                    if flat.num_rows() > 0 {
                        output.push(flat);
                    }
                }
            } else {
                break;
            }
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

            let start = rows_col.value_offsets()[0] as usize;
            let end = rows_col.value_offsets()[1] as usize;

            for i in start..end {
                let d = dist_component.value(i);
                let scalar = ScalarValue::try_from_array(row_component, i)?;
                candidates.push(Candidate {
                    dist: d,
                    data: scalar,
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

        // Build rows from scalars
        // inner struct: Struct<row, dist>
        let row_fields = self.result_schema.fields().clone();
        let combined_fields = vec![
            Field::new("row", DataType::Struct(row_fields.clone()), false),
            Field::new("dist", DataType::Float64, false),
        ];

        let mut dist_builder = Float64Builder::new();
        let mut row_scalars = Vec::new();

        for c in candidates {
            dist_builder.append_value(c.dist);
            row_scalars.push(c.data.clone());
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
        let scalars: Vec<ScalarValue> = candidates.iter().map(|c| c.data.clone()).collect();
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
