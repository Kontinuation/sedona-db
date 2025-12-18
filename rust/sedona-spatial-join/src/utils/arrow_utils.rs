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

use arrow::array::{Array, BinaryViewArray, RecordBatch, StringViewArray};
use datafusion_common::Result;

/// Reconstruct `batch` to organize the payload buffers of each `StringViewArray` and
/// `BinaryViewArray` in sequential order by calling `gc()` on them.
///
/// Note this is a workaround until <https://github.com/apache/arrow-rs/issues/7185> is
/// available.
///
/// # Rationale
///
/// The `interleave` kernel does not reconstruct the inner buffers of view arrays by default,
/// leading to non-sequential payload locations. A single payload buffer might be shared by
/// multiple `RecordBatch`es or multiple rows in the same batch might reference scattered
/// locations in a large buffer.
///
/// When writing each batch to disk, the writer has to write all referenced buffers. This
/// causes extra disk reads and writes, and potentially execution failure (e.g. No space left
/// on device).
///
/// # Example
///
/// Before interleaving:
/// batch1 -> buffer1 (large)
/// batch2 -> buffer2 (large)
///
/// interleaved_batch -> buffer1 (sparse access)
///                   -> buffer2 (sparse access)
///
/// Then when spilling the interleaved batch, the writer has to write both buffer1 and buffer2
/// entirely, even if only a few bytes are used.
pub(crate) fn compact_batch(batch: RecordBatch) -> Result<RecordBatch> {
    let mut new_columns: Vec<Arc<dyn Array>> = Vec::with_capacity(batch.num_columns());
    let mut arr_mutated = false;

    for array in batch.columns() {
        if let Some(view_array) = array.as_any().downcast_ref::<StringViewArray>() {
            new_columns.push(Arc::new(view_array.gc()));
            arr_mutated = true;
        } else if let Some(view_array) = array.as_any().downcast_ref::<BinaryViewArray>() {
            new_columns.push(Arc::new(view_array.gc()));
            arr_mutated = true;
        } else {
            new_columns.push(Arc::clone(array));
        }
    }

    if arr_mutated {
        Ok(RecordBatch::try_new(batch.schema(), new_columns)?)
    } else {
        Ok(batch)
    }
}
