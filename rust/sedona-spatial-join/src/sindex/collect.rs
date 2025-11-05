use sedona_expr::statistics::GeoStatistics;

pub(crate) struct BuildPartition {
    pub build_side_batch_stream: SendableBuildSideBatchStream,
    pub geo_statistics: GeoStatistics,
}

mod build_side_batch;
mod build_side_batch_stream;
mod build_side_collector;
mod spill;

pub(crate) use build_side_batch::BuildSideBatch;
pub(crate) use build_side_batch_stream::{BuildSideBatchStream, SendableBuildSideBatchStream};
pub(crate) use build_side_collector::{BuildSideBatchesCollector, CollectBuildSideMetrics};
