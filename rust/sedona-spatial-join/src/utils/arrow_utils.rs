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

use arrow::array::{make_array, Array, ArrayData, BinaryViewArray, RecordBatch, StringViewArray};
use arrow_array::ArrayRef;
use arrow_schema::{ArrowError, DataType};
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
        let (new_array, mutated) = compact_array(Arc::clone(array))?;
        new_columns.push(new_array);
        arr_mutated |= mutated;
    }

    if arr_mutated {
        Ok(RecordBatch::try_new(batch.schema(), new_columns)?)
    } else {
        Ok(batch)
    }
}

fn compact_array(array: ArrayRef) -> Result<(ArrayRef, bool)> {
    if let Some(view_array) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok((Arc::new(view_array.gc()), true));
    }
    if let Some(view_array) = array.as_any().downcast_ref::<BinaryViewArray>() {
        return Ok((Arc::new(view_array.gc()), true));
    }

    // Fast path for non-nested arrays
    if !array.data_type().is_nested() {
        return Ok((array, false));
    }

    // For nested arrays (Struct/List/Map/Dictionary/etc.), recurse into children via ArrayData.
    let data = array.to_data();
    if data.child_data().is_empty() {
        return Ok((array, false));
    }

    let mut mutated = false;
    let mut new_child_data = Vec::with_capacity(data.child_data().len());
    for child in data.child_data().iter() {
        let child_array = make_array(child.clone());
        let (new_child_array, child_mutated) = compact_array(child_array)?;
        mutated |= child_mutated;
        new_child_data.push(new_child_array.to_data());
    }

    if !mutated {
        return Ok((array, false));
    }

    // Rebuild this array with identical buffers/nulls but replaced child_data.
    let mut builder = data.into_builder();
    builder = builder.child_data(new_child_data);
    let new_data = builder.build()?;
    Ok((make_array(new_data), true))
}

/// Estimate the in-memory size of a given RecordBatch. This function estimates the
/// size as if the underlying buffers were copied to somewhere else and not shared.
pub(crate) fn get_record_batch_memory_size(batch: &RecordBatch) -> Result<usize> {
    let mut total_size = 0;

    for array in batch.columns() {
        let array_data = array.to_data();
        total_size += count_array_data_memory_size(&array_data)?;
    }

    Ok(total_size)
}

/// Estimate the in-memory size of a given Arrow array. This function estimates the
/// size as if the underlying buffers were copied to somewhere else and not shared.
pub(crate) fn get_array_memory_size(array: &ArrayRef) -> Result<usize> {
    let array_data = array.to_data();
    let size = count_array_data_memory_size(&array_data)?;
    Ok(size)
}

/// The maximum number of bytes that can be stored inline in a byte view.
///
/// See [`ByteView`] and [`GenericByteViewArray`] for more information on the
/// layout of the views.
///
/// [`GenericByteViewArray`]: https://docs.rs/arrow/latest/arrow/array/struct.GenericByteViewArray.html
pub const MAX_INLINE_VIEW_LEN: u32 = 12;

/// Count the memory usage of `array_data` and its children recursively.
fn count_array_data_memory_size(array_data: &ArrayData) -> core::result::Result<usize, ArrowError> {
    Ok(get_binary_view_value_size(array_data)? + array_data.get_slice_memory_size()?)
}

fn get_binary_view_value_size(array_data: &ArrayData) -> Result<usize, ArrowError> {
    let mut result: usize = 0;
    let array_data_type = array_data.data_type();

    if matches!(array_data_type, DataType::BinaryView | DataType::Utf8View) {
        // The views buffer contains length view structures with the following layout:
        // https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-view-layout
        //
        // * Short strings, length <= 12
        // | Bytes 0-3  | Bytes 4-15                            |
        // |------------|---------------------------------------|
        // | length     | data (padded with 0)                  |
        //
        // * Long strings, length > 12
        // | Bytes 0-3  | Bytes 4-7  | Bytes 8-11 | Bytes 12-15 |
        // |------------|------------|------------|-------------|
        // | length     | prefix     | buf. index | offset      |
        let views = &array_data.buffer::<u128>(0)[..array_data.len()];
        result = views
            .iter()
            .map(|v| {
                let len = *v as u32;
                if len > MAX_INLINE_VIEW_LEN {
                    len as usize
                } else {
                    0
                }
            })
            .sum();
    }

    for child in array_data.child_data() {
        result += get_binary_view_value_size(child)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{builder::StringViewBuilder, BooleanArray};
    use arrow_array::{BinaryViewArray, StringViewArray, StructArray};
    use arrow_schema::{DataType, Field};
    use std::sync::Arc;

    #[test]
    fn test_string_view_array_memory_size() {
        let array = StringViewArray::from(vec![
            "short",                                               // Inline
            "Long string that is definitely longer than 12 bytes", // 51 bytes
        ]);
        let array_ref: ArrayRef = Arc::new(array);
        let size = get_array_memory_size(&array_ref).unwrap();
        // Views: 2 * 16 = 32 bytes
        // Data: 51 bytes
        // Total: 83 bytes
        assert_eq!(size, 83);
    }

    #[test]
    fn test_binary_view_array_memory_size() {
        let array = BinaryViewArray::from(vec![
            "short".as_bytes(),
            "Long string that is definitely longer than 12 bytes".as_bytes(),
        ]);
        let array_ref: ArrayRef = Arc::new(array);
        let size = get_array_memory_size(&array_ref).unwrap();
        assert_eq!(size, 83);
    }

    #[test]
    fn test_struct_array_with_view_memory_size() {
        let string_view_array = StringViewArray::from(vec![
            "short",
            "Long string that is definitely longer than 12 bytes",
        ]);
        let boolean_array = arrow_array::BooleanArray::from(vec![true, false]);

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("a", DataType::Utf8View, false)),
                Arc::new(string_view_array) as ArrayRef,
            ),
            (
                Arc::new(Field::new("b", DataType::Boolean, false)),
                Arc::new(boolean_array) as ArrayRef,
            ),
        ]);

        let array_ref: ArrayRef = Arc::new(struct_array);
        let size = get_array_memory_size(&array_ref).unwrap();
        // 83 (StringView) + 1 (Boolean values) = 84
        assert_eq!(size, 84);
    }

    #[test]
    fn test_sliced_view_array_memory_size() {
        let array = StringViewArray::from(vec![
            "short",
            "Long string that is definitely longer than 12 bytes",
            "Another long string to make buffer larger",
        ]);
        let sliced = array.slice(0, 2);
        let sliced_ref: ArrayRef = Arc::new(sliced);
        let size = get_array_memory_size(&sliced_ref).unwrap();
        // Views: 2 * 16 = 32
        // Data used: 51 ("Long string...")
        // Total: 83
        assert_eq!(size, 83);
    }

    #[test]
    fn test_compact_batch_recurses_into_struct() {
        let n = 256;
        let long = "x".repeat(2048);

        let mut builder = StringViewBuilder::with_capacity(n);
        for i in 0..n {
            builder.append_value(&format!("batch0_{i}_{long}"));
        }
        let string_view_array: ArrayRef = Arc::new(builder.finish());
        let boolean_array: ArrayRef = Arc::new(BooleanArray::from(vec![true; n]));
        let struct_fields = vec![
            Arc::new(Field::new("a", DataType::Utf8View, false)),
            Arc::new(Field::new("b", DataType::Boolean, false)),
        ];
        let struct_array = StructArray::from(vec![
            (
                Arc::clone(&struct_fields[0]),
                Arc::clone(&string_view_array),
            ),
            (Arc::clone(&struct_fields[1]), Arc::clone(&boolean_array)),
        ]);

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "s",
            DataType::Struct(struct_fields.into()),
            false,
        )]));
        let batch0 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(struct_array) as ArrayRef],
        )
        .unwrap();
        let sliced = batch0.slice(0, 1);

        let before = sliced.get_array_memory_size();
        let compacted = compact_batch(sliced.clone()).unwrap();
        let after = compacted.get_array_memory_size();

        assert_eq!(sliced.schema(), compacted.schema());
        assert_eq!(sliced.num_rows(), compacted.num_rows());
        assert!(
            after < before,
            "expected compaction to reduce memory: before={before}, after={after}"
        );
    }
}
