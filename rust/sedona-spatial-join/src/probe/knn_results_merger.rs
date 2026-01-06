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
use std::io::{BufReader, Seek, SeekFrom};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, Float64Array, Float64Builder, ListArray, ListBuilder, RecordBatch,
    StructArray, StructBuilder,
};
use arrow::compute::interleave;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use datafusion::config::SpillCompression;
use datafusion_common::Result;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_plan::metrics::SpillMetrics;
use parking_lot::Mutex;
use sedona_common::sedona_internal_err;
use tempfile::tempfile;

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
    previous: Option<File>,
    /// File to write results for current (0..N) partitions
    current: Option<File>,
    /// Only used when writing to current
    current_writer: Option<FileWriter<File>>,
    /// Reader for previous file (re-created for each batch ingestion?)
    previous_reader: Option<FileReader<BufReader<File>>>,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    dist: f64,
    source_idx: usize, // Index in the value array (0 for new, 1 for old)
    row_idx: usize,    // Index within the source array
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
                previous: None,
                current: None,
                current_writer: None,
                previous_reader: None,
            }),
        }
    }

    pub fn is_single_partitioned(&self) -> bool {
        let state = self.state.lock();
        state.previous.is_none() && state.current.is_none()
    }

    fn create_spill_schema(result_schema: SchemaRef) -> SchemaRef {
        // Schema:
        // rows: List<Struct<row: Struct<...>, dist: Float64>>
        // unfiltered_dists: Float64

        // The inner "row" struct matches result_schema
        let row_field = Field::new(
            "row",
            DataType::Struct(result_schema.fields().clone()),
            false,
        );
        let dist_field = Field::new("dist", DataType::Float64, false);

        let struct_fields = vec![row_field, dist_field];
        let struct_type = DataType::Struct(struct_fields.into());

        // The "rows" column is a List of that struct
        let rows_field = Field::new(
            "rows",
            DataType::List(Arc::new(Field::new("item", struct_type, true))),
            false,
        );
        // "unfiltered_dists" tracks the k-th distance
        let unfiltered_dists_field = Field::new("unfiltered_dists", DataType::Float64, true);

        Arc::new(Schema::new(vec![rows_field, unfiltered_dists_field]))
    }

    /// Rotate the spill files.
    /// probing_last_index: true if we are about to probe the last partition.
    pub fn rotate(&self, probing_last_index: bool) -> Result<()> {
        let mut state = self.state.lock();

        // 1. Close current writer if exists
        if let Some(mut writer) = state.current_writer.take() {
            writer.finish()?;
        }

        // 2. Determine new 'previous'
        let new_previous = state.current.take();

        // 3. Drop old 'previous' (file will be deleted if tempfile)
        state.previous = new_previous;
        state.previous_reader = None;

        // 4. If previous exists, create reader
        if let Some(file) = &state.previous {
            let mut file_clone = file.try_clone()?;
            file_clone.seek(SeekFrom::Start(0))?;
            state.previous_reader = Some(FileReader::try_new(BufReader::new(file_clone), None)?);
        }

        // 5. Create new 'current' if not probing last index
        if !probing_last_index {
            let file = tempfile()?;
            let writer = FileWriter::try_new(file.try_clone()?, &self.spill_schema)?;
            state.current = Some(file);
            state.current_writer = Some(writer);
        }

        Ok(())
    }

    /// Prepare for the first partition (if not single partition)
    pub fn init_for_partition_0(&self, is_last: bool) -> Result<()> {
        let mut state = self.state.lock();
        if !is_last {
            let file = tempfile()?;
            let writer = FileWriter::try_new(file.try_clone()?, &self.spill_schema)?;
            state.current = Some(file);
            state.current_writer = Some(writer);
        }
        Ok(())
    }

    pub fn ingest(
        &self,
        joined_batch: RecordBatch,
        distances: Option<&[f64]>,
        probe_indices: &[u32],
        offset_in_partition: usize,
        probe_batch_size: usize,
    ) -> Result<Option<RecordBatch>> {
        if self.is_single_partitioned() {
            return Ok(Some(joined_batch));
        }

        let prev_batch = {
            let mut state = self.state.lock();
            if let Some(reader) = &mut state.previous_reader {
                if let Some(res) = reader.next() {
                    Some(res?)
                } else {
                    None
                }
            } else {
                None
            }
        };

        let Some(distances) = distances else {
            return sedona_internal_err!(
                "distances should not be None when running multi-partitioned KNN join"
            );
        };
        self.merge_and_process(
            joined_batch,
            distances,
            probe_indices,
            offset_in_partition,
            probe_batch_size,
            prev_batch,
        )
    }

    fn merge_and_process(
        &self,
        batch: RecordBatch,
        distances: &[f64],
        probe_indices: &[u32],
        offset_in_partition: usize,
        probe_batch_size: usize,
        prev_batch: Option<RecordBatch>,
    ) -> Result<Option<RecordBatch>> {
        // Prepare New Data Components
        let new_row_struct = StructArray::from(batch);
        let new_dist_array = Float64Array::from(distances.to_vec());

        // Prepare Old Data Components
        let (old_rows_list, old_row_struct, old_dist_array, _old_thresholds) =
            if let Some(prev) = &prev_batch {
                let rows_col = prev.column(0).as_list::<i32>();
                let values_struct = rows_col.values().as_struct();

                // Assume "rows" col structure: List<Struct<row: Struct, dist: f64>>
                let old_row_component = values_struct.column(0).as_struct();
                let old_dist_component = values_struct
                    .column(1)
                    .as_primitive::<arrow::datatypes::Float64Type>();
                let input_thresholds = prev
                    .column(1)
                    .as_primitive::<arrow::datatypes::Float64Type>();

                (
                    Some(rows_col),
                    Some(old_row_component),
                    Some(old_dist_component),
                    Some(input_thresholds),
                )
            } else {
                (None, None, None, None)
            };

        // Group new results by probe index
        let mut new_results_by_probe = vec![vec![]; probe_batch_size];
        for (i, &probe_idx) in probe_indices.iter().enumerate() {
            if (probe_idx as usize) < probe_batch_size {
                new_results_by_probe[probe_idx as usize].push(i);
            }
        }

        // Output builders
        let mut selection_indices: Vec<(usize, usize)> = Vec::new(); // (source_idx, row_idx)
        let mut list_offsets: Vec<i32> = Vec::with_capacity(probe_batch_size + 1);
        list_offsets.push(0);
        let mut current_offset = 0;
        let mut new_thresholds_builder = Float64Builder::new();

        // Iterate per probe row
        for i in 0..probe_batch_size {
            let mut candidates: Vec<Candidate> = Vec::with_capacity(self.k * 2);

            // 1. Add new candidates
            for &row_idx in &new_results_by_probe[i] {
                let d = distances[row_idx];
                candidates.push(Candidate {
                    dist: d,
                    source_idx: 0,
                    row_idx,
                });
            }

            // 2. Add old candidates
            if let Some(list_arr) = old_rows_list {
                if i < list_arr.len() {
                    let start = list_arr.value_offsets()[i] as usize;
                    let end = list_arr.value_offsets()[i + 1] as usize;
                    if let Some(dists) = old_dist_array {
                        for old_idx in start..end {
                            let d = dists.value(old_idx);
                            candidates.push(Candidate {
                                dist: d,
                                source_idx: 1,
                                row_idx: old_idx,
                            });
                        }
                    }
                }
            }

            // 3. Sort and keep top K
            // We want K nearest, so smallest distance first.
            candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            if candidates.len() > self.k {
                candidates.truncate(self.k);
            }

            // 4. Record selections
            for c in &candidates {
                selection_indices.push((c.source_idx, c.row_idx));
            }

            current_offset += candidates.len() as i32;
            list_offsets.push(current_offset);

            // 5. Update threshold
            if candidates.len() >= self.k {
                new_thresholds_builder.append_value(candidates.last().unwrap().dist);
            } else {
                new_thresholds_builder.append_null(); // Less than K results found so far
            }
        }

        // Apply Interleave to build combined arrays
        let arrays_for_row_struct: Vec<&dyn Array> = vec![
            &new_row_struct,
            old_row_struct
                .map(|x| x as &dyn Array)
                .unwrap_or(&new_row_struct),
        ];
        let arrays_for_dist: Vec<&dyn Array> = vec![
            &new_dist_array,
            old_dist_array
                .map(|x| x as &dyn Array)
                .unwrap_or(&new_dist_array),
        ];

        let final_rows_struct = interleave(&arrays_for_row_struct, &selection_indices)?;
        let final_dist_array = interleave(&arrays_for_dist, &selection_indices)?;

        // If is_last_partition, we return the flattened result.
        let mut state = self.state.lock();
        if let Some(writer) = &mut state.current_writer {
            // We define the ListArray for spill file.
            // Struct<row, dist>
            let result_fields = final_rows_struct.as_struct().fields().clone();

            let combined_fields = vec![
                Field::new("row", DataType::Struct(result_fields), false),
                Field::new("dist", DataType::Float64, false),
            ];

            let combined_struct = StructArray::try_new(
                combined_fields.into(),
                vec![Arc::new(final_rows_struct), Arc::new(final_dist_array)],
                None,
            )?;

            let list_field = Arc::new(Field::new(
                "item",
                combined_struct.data_type().clone(),
                true,
            ));
            let list_array = ListArray::try_new(
                list_field,
                arrow::buffer::OffsetBuffer::new(list_offsets.into()),
                Arc::new(combined_struct),
                None,
            )?;

            let new_thresholds = new_thresholds_builder.finish();

            let spill_batch = RecordBatch::try_new(
                self.spill_schema.clone(),
                vec![Arc::new(list_array), Arc::new(new_thresholds)],
            )?;

            // Write to current spill file without returning the result batch
            writer.write(&spill_batch)?;
            Ok(None)
        } else {
            // We are probing the last partition. Return the result directly without writing
            // them to the next spill file.

            // final_rows_struct is a StructArray containing the final rows in order.
            // We can convert it to RecordBatch directly.
            let struct_arr = final_rows_struct.as_struct();
            let batch = RecordBatch::from(struct_arr);
            Ok(Some(batch))
        }
    }
}
