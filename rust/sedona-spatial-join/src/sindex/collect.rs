use datafusion_execution::memory_pool::MemoryReservation;
use sedona_expr::statistics::GeoStatistics;

mod build_side_batch;
mod build_side_batch_stream;
mod build_side_collector;
mod spill;

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
