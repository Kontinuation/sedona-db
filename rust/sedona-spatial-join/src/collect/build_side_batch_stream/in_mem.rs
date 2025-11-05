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
    pin::Pin,
    task::{Context, Poll},
};

use datafusion_common::Result;

use crate::collect::{
    build_side_batch::BuildSideBatch, build_side_batch_stream::BuildSideBatchStream,
};

/// A build side batch stream that holds all batches in memory.
pub(crate) struct InMemoryBuildSideBatchStream {
    batches: VecDeque<BuildSideBatch>,
}

impl InMemoryBuildSideBatchStream {
    pub fn new(batches: Vec<BuildSideBatch>) -> Self {
        InMemoryBuildSideBatchStream {
            batches: VecDeque::from(batches),
        }
    }
}

impl BuildSideBatchStream for InMemoryBuildSideBatchStream {
    fn is_external(&self) -> bool {
        false
    }
}

impl futures::Stream for InMemoryBuildSideBatchStream {
    type Item = Result<BuildSideBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let front = self.get_mut().batches.pop_front();
        match front {
            Some(batch) => Poll::Ready(Some(Ok(batch))),
            None => Poll::Ready(None),
        }
    }
}
