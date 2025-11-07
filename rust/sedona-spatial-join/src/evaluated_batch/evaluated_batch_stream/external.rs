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
    pin::Pin,
    task::{Context, Poll},
};

use datafusion_common::{DataFusionError, Result};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::disk_manager::RefCountedTempFile;
use futures::FutureExt;

use crate::evaluated_batch::{
    evaluated_batch_stream::EvaluatedBatchStream, spill::SpillReader, EvaluatedBatch,
};

pub(crate) struct ExternalEvaluatedBatchStream {
    state: State,
}

enum State {
    UnInitialized(RefCountedTempFile),
    Opening(SpawnedTask<Result<SpillReader>>),
    Reading(SpawnedTask<(SpillReader, Option<Result<EvaluatedBatch>>)>),
    Finished,
}

impl ExternalEvaluatedBatchStream {
    pub fn try_new(spill_file: RefCountedTempFile) -> Result<Self> {
        Ok(Self {
            state: State::UnInitialized(spill_file),
        })
    }
}

impl EvaluatedBatchStream for ExternalEvaluatedBatchStream {
    fn is_external(&self) -> bool {
        true
    }
}

impl futures::Stream for ExternalEvaluatedBatchStream {
    type Item = Result<EvaluatedBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let self_mut = self.get_mut();

        loop {
            match std::mem::replace(&mut self_mut.state, State::Finished) {
                State::UnInitialized(spill_file) => {
                    let task =
                        SpawnedTask::spawn_blocking(move || SpillReader::try_new(&spill_file));
                    self_mut.state = State::Opening(task);
                }
                State::Opening(mut task) => match futures::ready!(task.poll_unpin(cx)) {
                    Err(e) => {
                        return Poll::Ready(Some(Err(DataFusionError::External(Box::new(e)))));
                    }
                    Ok(Err(e)) => {
                        return Poll::Ready(Some(Err(e)));
                    }
                    Ok(Ok(mut spill_reader)) => {
                        let task = SpawnedTask::spawn_blocking(move || {
                            let next_batch = spill_reader.next_batch();
                            (spill_reader, next_batch)
                        });
                        self_mut.state = State::Reading(task);
                    }
                },
                State::Reading(mut task) => match futures::ready!(task.poll_unpin(cx)) {
                    Err(e) => {
                        return Poll::Ready(Some(Err(DataFusionError::External(Box::new(e)))));
                    }
                    Ok((_, None)) => {
                        return Poll::Ready(None);
                    }
                    Ok((_, Some(Err(e)))) => {
                        return Poll::Ready(Some(Err(e)));
                    }
                    Ok((mut spill_reader, Some(Ok(batch)))) => {
                        let task = SpawnedTask::spawn_blocking(move || {
                            let next_batch = spill_reader.next_batch();
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
