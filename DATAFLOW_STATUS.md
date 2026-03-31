# Current Dataflow Status

Based on the current successful run in `b.log`, the dataflow is:

1. Rollout results are first `DataProto`, then written into TQ, and then split into per-sample `LazyDataProto` before entering the replay buffer.

   - `generate_scheduler.commit_to_replay.combined` -> `DataProto`
   - after TQ put -> `KVBatchMeta`
   - `generate_scheduler.commit_to_replay.sample` -> `LazyDataProto`

   This means TQ/lazy is currently active at the scheduler replay-storage layer.

2. When the scheduler collects a batch back from the replay buffer, it may concatenate `LazyDataProto` internally, but the object returned from `get_batch()` to `rlvr_pipeline` is already `DataProto`.

   - `generate_scheduler.collect.domain_batch` -> `LazyDataProto`
   - `rlvr.after_scheduler_get_batch` -> `DataProto`

   So the scheduler can be lazy internally, but the pipeline boundary is currently eager.

3. Inside `rlvr_pipeline`, the main path is currently eager `DataProto`.

   Confirmed by these logs:

   - `rlvr.after_scheduler_get_batch`
   - `rlvr.after_generate_output_concat`
   - `rlvr.after_ref_compute_log_probs`
   - `rlvr.after_actor_compute_log_probs_concat`
   - `rlvr.after_reward_concat`

   This means reorder/group_by/reward/advantage/union are currently operating on local eager `DataProto`, not on lazy handles.

4. In the currently successful path, remote worker dispatch is also sending `DataProto`, not `KVBatchMeta`.

   Confirmed by:

   - `decorator.prepare_for_remote.input` -> `DataProto`
   - `decorator.prepare_for_remote.output` -> `DataProto`

   Worker results are then collected back through `ObjectRefWrap` / `ObjectRef`, and finally merged into `DataProto` by `BatchData.concat()` or `DataProto.materialize_concat()`.

## Short Conclusion

The current effective status is:

1. TQ/lazy is active in the scheduler replay-storage layer.
2. The object returned into `rlvr_pipeline` is currently eager `DataProto`.
3. The main computation path in `rlvr_pipeline` is currently eager.
4. The currently successful remote dispatch path is also effectively eager `DataProto`.

So right now, TQ/lazy is helping mainly at the replay buffer / scheduler storage boundary, while the pipeline and worker compute path is still running in eager `DataProto` mode.
