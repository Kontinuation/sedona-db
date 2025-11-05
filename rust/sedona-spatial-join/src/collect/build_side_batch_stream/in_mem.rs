use std::{
    collections::VecDeque,
    pin::Pin,
    task::{Context, Poll},
};

use datafusion_common::Result;

use crate::collect::{
    build_side_batch::BuildSideBatch, build_side_batch_stream::BuildSideBatchStream,
};

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
