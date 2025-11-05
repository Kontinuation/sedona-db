use std::{
    pin::Pin,
    task::{Context, Poll},
};

use datafusion_common::Result;
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_physical_plan::SpillManager;

use crate::collect::{
    build_side_batch::BuildSideBatch, build_side_batch_stream::BuildSideBatchStream,
};

pub(crate) struct ExternalBuildSideBatchStream {
    // TODO: implement spilled batch stream
}

impl ExternalBuildSideBatchStream {
    pub fn try_new(spill_manager: SpillManager, spill_file: RefCountedTempFile) -> Result<Self> {
        let _stream = spill_manager.read_spill_as_stream(spill_file)?;
        todo!()
    }
}

impl BuildSideBatchStream for ExternalBuildSideBatchStream {
    fn is_external(&self) -> bool {
        true
    }
}

impl futures::Stream for ExternalBuildSideBatchStream {
    type Item = Result<BuildSideBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}
