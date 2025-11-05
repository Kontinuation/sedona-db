use std::pin::Pin;

use futures::Stream;

use crate::collect::build_side_batch::BuildSideBatch;
use datafusion_common::Result;

/// A stream that produces BuildSideBatch items. This stream may have purely in-memory or
/// out-of-core implementations. The type of the stream could be queried calling `is_external()`.
pub(crate) trait BuildSideBatchStream: Stream<Item = Result<BuildSideBatch>> {
    /// Returns true if this stream is an external stream, where batch data were spilled to disk.
    fn is_external(&self) -> bool;
}

pub(crate) type SendableBuildSideBatchStream = Pin<Box<dyn BuildSideBatchStream + Send>>;

pub(crate) mod in_mem;
