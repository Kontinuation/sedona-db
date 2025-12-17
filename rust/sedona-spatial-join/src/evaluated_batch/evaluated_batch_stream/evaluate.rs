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

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use datafusion_common::Result;
use datafusion_physical_plan::SendableRecordBatchStream;
use futures::{Stream, StreamExt};

use crate::evaluated_batch::{
    evaluated_batch_stream::{EvaluatedBatchStream, SendableEvaluatedBatchStream},
    EvaluatedBatch,
};
use crate::operand_evaluator::{EvaluatedGeometryArray, OperandEvaluator};

trait Evaluator: Unpin {
    fn evaluate(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray>;
}

struct BuildSideEvaluator {
    evaluator: Arc<dyn OperandEvaluator>,
}

impl Evaluator for BuildSideEvaluator {
    fn evaluate(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray> {
        self.evaluator.evaluate_build(batch)
    }
}

struct ProbeSideEvaluator {
    evaluator: Arc<dyn OperandEvaluator>,
}

impl Evaluator for ProbeSideEvaluator {
    fn evaluate(&self, batch: &RecordBatch) -> Result<EvaluatedGeometryArray> {
        self.evaluator.evaluate_probe(batch)
    }
}

/// Wraps a `SendableRecordBatchStream` and evaluates the probe-side geometry
/// expression eagerly so downstream consumers can operate on `EvaluatedBatch`s.
struct EvaluateRecordStream<E: Evaluator> {
    inner: SendableRecordBatchStream,
    evaluator: E,
}

impl<E: Evaluator> EvaluateRecordStream<E> {
    fn new(inner: SendableRecordBatchStream, evaluator: E) -> Self {
        Self { inner, evaluator }
    }
}

impl<E: Evaluator> EvaluatedBatchStream for EvaluateRecordStream<E> {
    fn is_external(&self) -> bool {
        false
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.inner.schema()
    }
}

impl<E: Evaluator> Stream for EvaluateRecordStream<E> {
    type Item = Result<EvaluatedBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let self_mut = self.get_mut();
        match self_mut.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let geom_array = self_mut.evaluator.evaluate(&batch)?;
                let evaluated = EvaluatedBatch { batch, geom_array };
                Poll::Ready(Some(Ok(evaluated)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Returns a `SendableEvaluatedBatchStream` that eagerly evaluates the build-side
/// geometry expression for every incoming `RecordBatch`.
pub(crate) fn create_evaluated_build_stream(
    stream: SendableRecordBatchStream,
    evaluator: Arc<dyn OperandEvaluator>,
) -> SendableEvaluatedBatchStream {
    Box::pin(EvaluateRecordStream::new(
        stream,
        BuildSideEvaluator { evaluator },
    ))
}

/// Returns a `SendableEvaluatedBatchStream` that eagerly evaluates the probe-side
/// geometry expression for every incoming `RecordBatch`.
pub(crate) fn create_evaluated_probe_stream(
    stream: SendableRecordBatchStream,
    evaluator: Arc<dyn OperandEvaluator>,
) -> SendableEvaluatedBatchStream {
    Box::pin(EvaluateRecordStream::new(
        stream,
        ProbeSideEvaluator { evaluator },
    ))
}
