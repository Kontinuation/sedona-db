# Design Document: Refactoring KNN Join in SpatialJoinStream

## 1. Problem Statement
The current implementation of partitioned KNN join introduces significant complexity into `SpatialJoinStream` (in `stream.rs`). It adds multiple KNN-specific fields (spill files, writers, schemas) and modifies the state machine logic directly, violating the Single Responsibility Principle. The goal is to encapsulate KNN-specific logic (spilling, merging, result generation) into a dedicated abstraction, keeping `stream.rs` clean and focused on the high-level join orchestration.

## 2. Proposed Solution
We will introduce a `ProbeStreamProvider` abstraction (implemented as an enum) that unifies the handling of probe streams for both standard spatial joins and KNN joins. 

- **Standard Spatial Join**: Uses `PartitionedProbeStreamProvider` (existing).
- **KNN Join**: Uses a new `KNNProbeStreamProvider` (to be created).

`SpatialJoinStream` will interact with this provider interface, delegating the details of stream fetching, result processing, and spilling to the provider.

## 3. Detailed Design

### 3.1. `ProbeStreamProvider` Enum
We will define an enum `ProbeStreamProvider` in `src/probe/mod.rs` or `src/stream.rs` (or a new module) that wraps the two implementations.

```rust
pub(crate) enum ProbeStreamProvider {
    Standard(PartitionedProbeStreamProvider),
    Knn(KNNProbeStreamProvider),
}
```

This enum will expose the following interface methods:

1.  **`stream_for`**: Returns the probe stream for a given partition.
    ```rust
    fn stream_for(&self, partition: SpatialPartition) -> Result<SendableEvaluatedBatchStream>;
    ```
    - *Standard*: Delegates to `PartitionedProbeStreamProvider::stream_for`.
    - *KNN*: Handles the logic to force `SpatialPartition::Multi` (or `Regular(0)` for single-partition) regardless of the requested partition, as KNN needs to probe all rows against every index partition.

2.  **`prepare_for_partition`**: Called when `SpatialJoinStream` moves to a new build (index) partition.
    ```rust
    fn prepare_for_partition(&mut self, partition_id: usize) -> Result<()>;
    ```
    - *Standard*: No-op.
    - *KNN*: Rotates spill files. The `next` spill file from the previous round becomes the `prev` input stream for the current round. Creates a new `next` spill file for the current round.

3.  **`process_batch_result`**: Processes a result batch produced by the `SpatialJoinBatchIterator`.
    ```rust
    fn process_batch_result(
        &mut self, 
        batch: RecordBatch, 
        probe_indices: Option<Vec<u32>>,
        is_last_partition: bool
    ) -> Result<Option<RecordBatch>>;
    ```
    - *Standard*: Returns `Ok(Some(batch))`. The stream immediately yields this batch.
    - *KNN*: 
        - Accumulates the batch and `probe_indices`.
        - If the probe batch is complete (logic handled internally or via flag), it performs the **Merge & Spill** operation:
            - Merges current results with results from `prev` spill file (if any).
            - Keeps top-K for each probe row.
            - If `!is_last_partition`: Writes the refined results to `next` spill file. Returns `Ok(None)`.
            - If `is_last_partition`: Returns `Ok(Some(final_batch))`.

### 3.2. `KNNProbeStreamProvider`
A new struct `KNNProbeStreamProvider` will be created (likely in `src/probe/knn_stream_provider.rs`). It will encapsulate all the fields previously added to `SpatialJoinStream`:

- `knn_spill_file`, `knn_next_spill_file`
- `knn_prev_stream`, `knn_next_writer`
- `knn_schema`
- `knn_current_probe_batch_results`
- `partitioned_provider`: The underlying `PartitionedProbeStreamProvider` to fetch raw probe batches.

### 3.3. `SpatialJoinStream` Refactoring

1.  **Field Cleanup**: Remove all `knn_*` fields. Replace `probe_stream_provider` (currently `Option<PartitionedProbeStreamProvider>`) with `Option<ProbeStreamProvider>`.
2.  **State Machine**:
    - **`WaitBuildIndex` / `PrepareForNextPartition`**: Call `provider.prepare_for_partition(partition_id)`.
    - **`FetchProbeBatch`**: Call `provider.stream_for(...)`.
    - **`ProcessProbeBatch`**: 
        - When `iterator.next_batch()` returns a result, call `provider.process_batch_result(...)`.
        - If it returns `Some(batch)`, yield it.
        - If it returns `None`, continue (loop).
    - **Remove `YieldKnnResults`**: Since `process_batch_result` returns the final batch during the last partition processing, we don't need a separate state to drain a spill file.

### 3.4. `SpatialJoinBatchIterator` Modification
We need to slightly modify `SpatialJoinBatchIterator::next_batch` to return `Result<Option<(RecordBatch, Option<Vec<u32>>)>>`.
- The `Option<Vec<u32>>` will contain the probe indices (row IDs) corresponding to the matched rows.
- This is necessary for KNN to group matches by probe row during the merge phase.
- For Standard join, this can be `None`.

## 4. Interaction Flow (KNN Scenario)

1.  **Initialization**: `SpatialJoinStream` creates `KNNProbeStreamProvider`.
2.  **Partition 0**:
    - `prepare_for_partition(0)`: Provider initializes first spill file (if needed).
    - `stream_for(...)`: Provider returns probe stream (all rows).
    - `ProcessProbeBatch`: 
        - Iterator produces matches.
        - `process_batch_result(..., is_last=false)`: Provider merges (no prev spill), keeps top-K, writes to Spill 1. Returns `None`.
3.  **Partition 1**:
    - `prepare_for_partition(1)`: Provider rotates: Spill 1 -> Prev Input. Creates Spill 2.
    - `stream_for(...)`: Provider returns probe stream again.
    - `ProcessProbeBatch`:
        - Iterator produces matches.
        - `process_batch_result(..., is_last=false)`: Provider merges with Spill 1, keeps top-K, writes to Spill 2. Returns `None`.
4.  **Last Partition (N)**:
    - `prepare_for_partition(N)`: Provider rotates: Spill N-1 -> Prev Input.
    - `stream_for(...)`: Provider returns probe stream.
    - `ProcessProbeBatch`:
        - Iterator produces matches.
        - `process_batch_result(..., is_last=true)`: Provider merges with Spill N-1, keeps top-K. **Returns `Some(final_batch)`**.
    - `SpatialJoinStream` yields the batch.

## 5. Benefits
- **Clean `stream.rs`**: No KNN specific fields or complex state transitions.
- **Encapsulation**: KNN logic is isolated.
- **Extensibility**: `ProbeStreamProvider` pattern can support other join types or strategies in the future.
- **Efficiency**: Reuses existing `PartitionedProbeStreamProvider` for data fetching.

## 6. Multi-Partitioned KNN Join Design

### 6.1. Partitioning Strategy
- **Indexed Side (Build Side)**: The object data is partitioned (e.g., using round-robin partitioning) into $N$ partitions. Each partition is indexed independently using a spatial index (e.g., R-Tree).
- **Probe Side (Query Side)**: The query data is treated as a single logical stream (Multi-partition). It is not spatially partitioned to match the build side. Instead, the entire probe stream is processed against *each* of the $N$ build partitions sequentially.

### 6.2. Execution Flow
The join is executed as a loop over the build partitions:

1.  **Initialization**: The `SpatialJoinStream` initializes the `KNNProbeStreamProvider`.
2.  **Iteration**: For each build partition $P_i$ (where $i$ ranges from $0$ to $N-1$):
    - **Load Index**: The spatial index for $P_i$ is built or loaded.
    - **Stream Probe Data**: The probe stream is fetched. For the first partition ($i=0$), this comes from the original input. For subsequent partitions ($i>0$), this comes from the spill file generated in the previous iteration ($i-1$).
    - **Local KNN Search**: Each probe row performs a KNN search against the index of $P_i$.
    - **Merge & Refine**:
        - The local top-$K$ matches from $P_i$ are merged with the current global top-$K$ matches (carried over from previous iterations).
        - The merged list is sorted by distance, and only the top $K$ are retained.
    - **Spill**: The updated top-$K$ matches are written to a new spill file (the "next" spill).
3.  **Finalization**:
    - After processing the last partition ($P_{N-1}$), the results in the final spill file represent the true global top-$K$ neighbors for each probe row.
    - These results are yielded as the output of the join.

### 6.3. Spilling and Merging Mechanism
To handle memory constraints and ensure correctness across partitions:

- **Spill Files**: Two spill files are maintained: `current_input` and `next_output`.
- **Rotation**: At the end of each partition iteration, `next_output` becomes `current_input` for the next iteration, and a new `next_output` is created.
- **Merge Logic**:
    - Input: A stream of `(ProbeRow, CurrentTopKMatches)` from the previous iteration.
    - Process: `ProbeRow` queries the current index partition to find `LocalMatches`.
    - Output: `Merge(CurrentTopKMatches, LocalMatches) -> NewTopKMatches`.
    - The `NewTopKMatches` are written to the output spill file.

This design ensures that we find the correct global nearest neighbors by exhaustively searching all partitions, while keeping memory usage bounded by spilling intermediate states.
