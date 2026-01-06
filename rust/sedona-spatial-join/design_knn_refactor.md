# Design Document: Refactoring KNN Join in SpatialJoinStream

## 1. Problem Statement
The current implementation of partitioned KNN join introduces significant complexity into `SpatialJoinStream` (in `stream.rs`). It adds multiple KNN-specific fields (spill files, writers, schemas) and modifies the state machine logic directly, violating the Single Responsibility Principle. The goal is to encapsulate KNN-specific logic (spilling, merging, result generation) into a dedicated abstraction, keeping `stream.rs` clean and focused on the high-level join orchestration.

## 2. Proposed Solution

We will introduce a `KNNResultsMerger` abstraction that handles KNN query results spilling and merging for KNN joins.

`SpatialJoinStream` will have a `Arc<KNNResultsMerger>` member.

We will use a Broadcast partitioner when running KNN join, which means that all the probe side data will be processed for each indexed partition. `KNNResultsMerger`'s
duty is to merge the results of probing each index partition to form the final results.

The spill files `KNNResultsMerger` manages contains currently k-nearest-so-far results. It should have a schema like this:

```
Struct([
    Field("rows",
        Array(
            Struct([
                Field("row": Struct(<the schema of the final KNN join result>)),
                Field("dist": Float64)
            ])
        )
    ),
    Field("unfiltered_dists", Array(Float64))
])
```

Basically, each row in the spill file is an array of k-nearest-so-far results we have seen. We call it "k-nearest-so-far" because we have only processed the
previous K-1 indexed partitions (assuming the current indexed partition is K). The records in the current indexed partition may be more near than what have seen
in the previous K-1 partitions. Certainly, after processing all the indexed partitions, we'll get the final k-nearest results.

Each `KNNResultsMerger` should maintain 2 spill files:

- `previous`: The k-nearest-so-far results for K-1 partitions, it is read only for current round.
- `current`: The spill files to be written for k-nearest-so-far results for K partitions.

Once we processed a indexed partition, we rotate the spill files: `current` is now `previous`, and we create a new file for writting `current`.

## 3. Coarse Directions to Implementation

### Changing the query_knn method of `SpatialIndex`

`SpatialIndex::query_knn` does not expose the actual distances computed for the k nearest neighbors. We need to add a new output parameter for this.
This could be something like `Option<&mut Vec<f64>>` or something. When it is none, it will skip exposing the computed distances. This will eliminate unnecessary
computations for single-indexed-partition KNN join.

### Passing the `KNNResultsMerger` to `SpatialJoinBatchIterator`.

`SpatialJoinBatchIterator` will hold a reference to `KNNResultsMerger` (could be an Arc of it), and it will feed the locally joined batch into `KNNResultsMerger`
to either write the merged k-nearest-so-far into the current spill file, or produce the k-nearest results to the caller.

`KNNResultsMerger` needs to have a state for understanding if we are probing the last indexed partition, and also have a method `ingest` or something (you can come up with a better name) for ingesting local result and return a `Option<RecordBatch>`:

- When we are not probing the last indexed partition, write merged results into `current` spill file and return `None`
- When we are probing the last indexed partition, directly return merged result as `Some`.

`KNNResultsMerger` also needs to have a `rotate(probing_last_index: bool)` method for rotating the `previous` and `current`. If we are probing the last index,
we can simply skip creating `current`.

There's a special case: when we run a fully in-memory KNN join where the first indexed partition is the last one, the `previous` and `current` spill files in
`KNNResultsMerger` are all `None`. `KNNResultsMerger::ingest` could simply return the incoming batch as is. This gracefully mimics the behavior of fully in memory
KNN join.

### Filtering of KNN join results

Please note that there's a filtering before assembling the KNN join results in `SpatialJoinBatchIterator::produce_result_batch`. If we don't do any special handling for it. partitioned KNN join could yield more results than single partitioned KNN join.

This is where `unfiltered_dists` field comes into play. We prune rows with `dist > unfiltered_dists` before producing the actual results.

## 4. Important Things to Note

1. We don't need to write spill file for the last indexed partition, we can simply return the k-nearest-results as the final join result
2. KNN joins are all inner joins. Any existing code for producing unmatched results for outer joins does not need to be adapted to support KNN join.
3. The `KNNResultsMerger` needs to handle the single partition case gracefully: if there's only one indexed probe partition (fully in-memory spatial join), we should
   effectively return the batch to ingest as is.
4. `KNNResultsMerger` should be implemented in its own file. It can be in the `src/probe` directory.
