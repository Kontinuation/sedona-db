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

use datafusion_execution::memory_pool::MemoryReservation;
use sedona_expr::statistics::GeoStatistics;

mod build_side_batch;
mod build_side_batch_stream;
mod build_side_collector;

pub(crate) use build_side_batch::BuildSideBatch;
pub(crate) use build_side_batch_stream::SendableBuildSideBatchStream;
pub(crate) use build_side_collector::{BuildSideBatchesCollector, CollectBuildSideMetrics};

pub(crate) struct BuildPartition {
    pub build_side_batch_stream: SendableBuildSideBatchStream,
    pub geo_statistics: GeoStatistics,

    /// Memory reservation for tracking the memory usage of the build partition
    /// Cleared on `BuildPartition` drop
    pub reservation: MemoryReservation,
}
