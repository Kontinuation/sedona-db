use std::{
    pin::Pin,
    task::{Context, Poll},
};

use datafusion_common::Result;
use futures::{Stream, StreamExt};

use crate::evaluated_batch::{
    evaluated_batch_stream::{EvaluatedBatchStream, SendableEvaluatedBatchStream},
    EvaluatedBatch,
};
use crate::probe::ProbeStreamMetrics;

pub(crate) struct NonPartitionedStream {
    inner: SendableEvaluatedBatchStream,
    metrics: ProbeStreamMetrics,
}

impl NonPartitionedStream {
    pub fn new(inner: SendableEvaluatedBatchStream, metrics: ProbeStreamMetrics) -> Self {
        Self { inner, metrics }
    }
}

impl Stream for NonPartitionedStream {
    type Item = Result<EvaluatedBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                self.metrics.probe_input_batches.add(1);
                self.metrics.probe_input_rows.add(batch.num_rows());
                Poll::Ready(Some(Ok(batch)))
            }
            other => other,
        }
    }
}

impl EvaluatedBatchStream for NonPartitionedStream {
    fn is_external(&self) -> bool {
        self.inner.as_ref().get_ref().is_external()
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.inner.schema()
    }
}
