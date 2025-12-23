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

use std::{
    collections::VecDeque,
    iter,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use datafusion_common::{DataFusionError, Result};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::{
    disk_manager::RefCountedTempFile, RecordBatchStream, SendableRecordBatchStream,
};
use datafusion_physical_plan::stream::RecordBatchReceiverStreamBuilder;
use futures::{FutureExt, StreamExt};
use pin_project_lite::pin_project;

use crate::evaluated_batch::{
    evaluated_batch_stream::EvaluatedBatchStream,
    spill::{spilled_batch_to_evaluated_batch, SpillReader},
    EvaluatedBatch,
};

const RECORD_BATCH_CHANNEL_CAPACITY: usize = 2;

pin_project! {
    pub(crate) struct ExternalEvaluatedBatchStream {
        #[pin]
        inner: RecordBatchToEvaluatedStream,
    }
}

pin_project! {
    struct RecordBatchToEvaluatedStream {
        #[pin]
        inner: SendableRecordBatchStream,
    }
}

pub(crate) struct ExternalRecordBatchStream {
    schema: SchemaRef,
    state: State,
    spill_files: VecDeque<Arc<RefCountedTempFile>>,
}

enum State {
    AwaitingFile,
    Opening(SpawnedTask<Result<SpillReader>>),
    Reading(SpawnedTask<(SpillReader, Option<Result<RecordBatch>>)>),
    Finished,
}

impl ExternalRecordBatchStream {
    pub fn try_from_spill_files<I>(spill_files: I) -> Result<Self>
    where
        I: IntoIterator<Item = Arc<RefCountedTempFile>>,
    {
        let spill_files = spill_files.into_iter().collect::<VecDeque<_>>();
        let schema = resolve_stream_schema(&spill_files)?;
        Ok(Self {
            schema,
            state: State::AwaitingFile,
            spill_files,
        })
    }
}

impl ExternalEvaluatedBatchStream {
    pub fn try_from_spill_file(spill_file: Arc<RefCountedTempFile>) -> Result<Self> {
        Self::try_from_spill_files(iter::once(spill_file))
    }

    pub fn try_from_spill_files<I>(spill_files: I) -> Result<Self>
    where
        I: IntoIterator<Item = Arc<RefCountedTempFile>>,
    {
        let record_stream = ExternalRecordBatchStream::try_from_spill_files(spill_files)?;
        Self::from_record_batch_stream(Box::pin(record_stream))
    }

    fn from_record_batch_stream(record_stream: SendableRecordBatchStream) -> Result<Self> {
        let schema = record_stream.schema();
        let mut builder =
            RecordBatchReceiverStreamBuilder::new(schema, RECORD_BATCH_CHANNEL_CAPACITY);
        let tx = builder.tx();
        builder.spawn(async move {
            let mut record_stream = record_stream;
            while let Some(batch) = record_stream.next().await {
                let is_err = batch.is_err();
                if tx.send(batch).await.is_err() {
                    break;
                }
                if is_err {
                    break;
                }
            }
            Ok(())
        });

        let buffered = builder.build();
        Ok(Self {
            inner: RecordBatchToEvaluatedStream::new(buffered),
        })
    }
}

impl EvaluatedBatchStream for ExternalEvaluatedBatchStream {
    fn is_external(&self) -> bool {
        true
    }

    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

impl futures::Stream for ExternalEvaluatedBatchStream {
    type Item = Result<EvaluatedBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().inner.poll_next(cx)
    }
}

impl RecordBatchToEvaluatedStream {
    fn new(inner: SendableRecordBatchStream) -> Self {
        Self { inner }
    }

    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

impl futures::Stream for RecordBatchToEvaluatedStream {
    type Item = Result<EvaluatedBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                Poll::Ready(Some(spilled_batch_to_evaluated_batch(batch)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for ExternalRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl futures::Stream for ExternalRecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let self_mut = self.get_mut();

        loop {
            match &mut self_mut.state {
                State::AwaitingFile => match self_mut.spill_files.pop_front() {
                    Some(spill_file) => {
                        let task =
                            SpawnedTask::spawn_blocking(move || SpillReader::try_new(&spill_file));
                        self_mut.state = State::Opening(task);
                    }
                    None => {
                        self_mut.state = State::Finished;
                        return Poll::Ready(None);
                    }
                },
                State::Opening(task) => match futures::ready!(task.poll_unpin(cx)) {
                    Err(e) => {
                        self_mut.state = State::Finished;
                        return Poll::Ready(Some(Err(DataFusionError::External(Box::new(e)))));
                    }
                    Ok(Err(e)) => {
                        self_mut.state = State::Finished;
                        return Poll::Ready(Some(Err(e)));
                    }
                    Ok(Ok(mut spill_reader)) => {
                        let task = SpawnedTask::spawn_blocking(move || {
                            let next_batch = spill_reader.next_raw_batch();
                            (spill_reader, next_batch)
                        });
                        self_mut.state = State::Reading(task);
                    }
                },
                State::Reading(task) => match futures::ready!(task.poll_unpin(cx)) {
                    Err(e) => {
                        self_mut.state = State::Finished;
                        return Poll::Ready(Some(Err(DataFusionError::External(Box::new(e)))));
                    }
                    Ok((_, None)) => {
                        self_mut.state = State::AwaitingFile;
                        continue;
                    }
                    Ok((_, Some(Err(e)))) => {
                        self_mut.state = State::Finished;
                        return Poll::Ready(Some(Err(e)));
                    }
                    Ok((mut spill_reader, Some(Ok(batch)))) => {
                        let task = SpawnedTask::spawn_blocking(move || {
                            let next_batch = spill_reader.next_raw_batch();
                            (spill_reader, next_batch)
                        });
                        self_mut.state = State::Reading(task);
                        return Poll::Ready(Some(Ok(batch)));
                    }
                },
                State::Finished => {
                    return Poll::Ready(None);
                }
            }
        }
    }
}

fn resolve_stream_schema(spill_files: &VecDeque<Arc<RefCountedTempFile>>) -> Result<SchemaRef> {
    if let Some(file) = spill_files.front() {
        let reader = SpillReader::try_new(file)?;
        Ok(reader.schema())
    } else {
        Ok(Arc::new(Schema::empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluated_batch::spill::SpillWriter;
    use crate::operand_evaluator::EvaluatedGeometryArray;
    use arrow_array::{Array, ArrayRef, BinaryArray, Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::config::SpillCompression;
    use datafusion_common::{Result, ScalarValue};
    use datafusion_execution::runtime_env::RuntimeEnv;
    use datafusion_expr::ColumnarValue;
    use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, SpillMetrics};
    use futures::StreamExt;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY};
    use std::sync::Arc;

    fn create_test_runtime_env() -> Result<Arc<RuntimeEnv>> {
        Ok(Arc::new(RuntimeEnv::default()))
    }

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_record_batch(start_id: i32) -> Result<RecordBatch> {
        let schema = create_test_schema();
        let id_array = Arc::new(Int32Array::from(vec![start_id, start_id + 1, start_id + 2]));
        let name_array = Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob"), None]));
        RecordBatch::try_new(schema, vec![id_array, name_array]).map_err(|e| e.into())
    }

    fn create_test_geometry_array() -> Result<(ArrayRef, SedonaType)> {
        // Create WKB encoded points (simple binary data for testing)
        let point1_wkb: Vec<u8> = vec![
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64,
        ];
        let point2_wkb: Vec<u8> = vec![
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 0, 16, 64,
        ];
        let point3_wkb: Vec<u8> = vec![
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 64, 0, 0, 0, 0, 0, 0, 24, 64,
        ];

        let sedona_type = WKB_GEOMETRY;
        let geom_array: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(point1_wkb.as_slice()),
            Some(point2_wkb.as_slice()),
            Some(point3_wkb.as_slice()),
        ]));

        Ok((geom_array, sedona_type))
    }

    fn create_test_evaluated_batch(start_id: i32) -> Result<EvaluatedBatch> {
        let batch = create_test_record_batch(start_id)?;
        let (geom_array, sedona_type) = create_test_geometry_array()?;
        let mut geom_array = EvaluatedGeometryArray::try_new(geom_array, &sedona_type)?;

        // Add distance as a scalar value
        geom_array.distance = Some(ColumnarValue::Scalar(ScalarValue::Float64(Some(10.0))));

        Ok(EvaluatedBatch { batch, geom_array })
    }

    async fn create_spill_file_with_batches(num_batches: usize) -> Result<RefCountedTempFile> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let mut writer = SpillWriter::try_new(
            env,
            schema,
            &sedona_type,
            "test_external_stream",
            SpillCompression::Uncompressed,
            metrics,
            None,
        )?;

        for i in 0..num_batches {
            let batch = create_test_evaluated_batch((i * 3) as i32)?;
            writer.append(&batch)?;
        }

        writer.finish()
    }

    #[tokio::test]
    async fn test_external_stream_creation() -> Result<()> {
        let spill_file = create_spill_file_with_batches(1).await?;
        let stream = ExternalEvaluatedBatchStream::try_from_spill_file(Arc::new(spill_file))?;

        assert!(stream.is_external());

        Ok(())
    }

    #[tokio::test]
    async fn test_external_stream_single_batch() -> Result<()> {
        let spill_file = create_spill_file_with_batches(1).await?;
        let mut stream = ExternalEvaluatedBatchStream::try_from_spill_file(Arc::new(spill_file))?;

        let batch = stream.next().await.unwrap()?;
        assert_eq!(batch.num_rows(), 3);

        // Polling again should still return None
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_external_stream_large_number_of_batches() -> Result<()> {
        let num_batches = 100;
        let spill_file = create_spill_file_with_batches(num_batches).await?;
        let mut stream = ExternalEvaluatedBatchStream::try_from_spill_file(Arc::new(spill_file))?;

        let mut count = 0;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            assert_eq!(batch.num_rows(), 3);
            count += 1;
        }

        assert_eq!(count, num_batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_external_stream_is_external_flag() -> Result<()> {
        let spill_file = create_spill_file_with_batches(1).await?;
        let stream = ExternalEvaluatedBatchStream::try_from_spill_file(Arc::new(spill_file))?;

        // Verify the is_external flag returns true
        assert!(stream.is_external());

        Ok(())
    }

    #[tokio::test]
    async fn test_external_stream_concurrent_access() -> Result<()> {
        // Create multiple streams reading from different files
        let file1 = create_spill_file_with_batches(2).await?;
        let file2 = create_spill_file_with_batches(3).await?;

        let mut stream1 = ExternalEvaluatedBatchStream::try_from_spill_file(Arc::new(file1))?;
        let mut stream2 = ExternalEvaluatedBatchStream::try_from_spill_file(Arc::new(file2))?;

        // Read from both streams
        let batch1_1 = stream1.next().await.unwrap()?;
        let batch2_1 = stream2.next().await.unwrap()?;

        assert_eq!(batch1_1.num_rows(), 3);
        assert_eq!(batch2_1.num_rows(), 3);

        // Continue reading
        let batch1_2 = stream1.next().await.unwrap()?;
        let batch2_2 = stream2.next().await.unwrap()?;

        assert_eq!(batch1_2.num_rows(), 3);
        assert_eq!(batch2_2.num_rows(), 3);

        // Stream1 should be done, stream2 should have one more
        assert!(stream1.next().await.is_none());
        let batch2_3 = stream2.next().await.unwrap()?;
        assert_eq!(batch2_3.num_rows(), 3);
        assert!(stream2.next().await.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_external_stream_multiple_spill_files() -> Result<()> {
        let file1 = create_spill_file_with_batches(2).await?;
        let file2 = create_spill_file_with_batches(3).await?;
        let mut stream = ExternalEvaluatedBatchStream::try_from_spill_files(vec![
            Arc::new(file1),
            Arc::new(file2),
        ])?;

        let mut batches_read = 0;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            assert_eq!(batch.num_rows(), 3);
            batches_read += 1;
        }

        assert_eq!(batches_read, 5);

        Ok(())
    }

    #[tokio::test]
    async fn test_external_stream_empty_file() -> Result<()> {
        let spill_file = create_spill_file_with_batches(0).await?;
        let mut stream = ExternalEvaluatedBatchStream::try_from_spill_file(Arc::new(spill_file))?;

        // Should immediately return None
        let batch = stream.next().await;
        assert!(batch.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_external_stream_empty_spill_file_list() -> Result<()> {
        let mut stream = ExternalEvaluatedBatchStream::try_from_spill_files(Vec::<
            Arc<RefCountedTempFile>,
        >::new())?;

        assert!(stream.next().await.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_external_stream_preserves_data() -> Result<()> {
        let spill_file = create_spill_file_with_batches(1).await?;
        let mut stream = ExternalEvaluatedBatchStream::try_from_spill_file(Arc::new(spill_file))?;

        let batch = stream.next().await.unwrap()?;

        // Verify data preservation
        let id_array = batch
            .batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_array.value(0), 0);
        assert_eq!(id_array.value(1), 1);
        assert_eq!(id_array.value(2), 2);

        let name_array = batch
            .batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name_array.value(0), "Alice");
        assert_eq!(name_array.value(1), "Bob");
        assert!(name_array.is_null(2));

        // Verify geometry array
        assert_eq!(batch.geom_array.rects.len(), 3);

        // Verify distance
        match &batch.geom_array.distance {
            Some(ColumnarValue::Scalar(ScalarValue::Float64(Some(val)))) => {
                assert_eq!(*val, 10.0);
            }
            _ => panic!("Expected scalar distance value"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_external_stream_multiple_batches() -> Result<()> {
        let num_batches = 5;
        let spill_file = create_spill_file_with_batches(num_batches).await?;
        let mut stream = ExternalEvaluatedBatchStream::try_from_spill_file(Arc::new(spill_file))?;

        let mut batches_read = 0;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            assert_eq!(batch.num_rows(), 3);

            // Verify the ID starts at the expected value
            let id_array = batch
                .batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let expected_start_id = (batches_read * 3) as i32;
            assert_eq!(id_array.value(0), expected_start_id);

            batches_read += 1;
        }

        assert_eq!(batches_read, num_batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_external_stream_poll_after_completion() -> Result<()> {
        let spill_file = create_spill_file_with_batches(1).await?;
        let mut stream = ExternalEvaluatedBatchStream::try_from_spill_file(Arc::new(spill_file))?;

        // Read the batch
        let _ = stream.next().await.unwrap()?;

        // Should return None
        assert!(stream.next().await.is_none());

        // Polling again should still return None
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());

        Ok(())
    }
}
