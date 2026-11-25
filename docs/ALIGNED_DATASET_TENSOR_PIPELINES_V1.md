# Differentiable aligned data loading — pure Oreslang CPU P1

**Status (October 18, 2026): Proposed source implementation and host-only numerical evidence. No pinned Java/Graal or LLVM guest execution on this exact tree.**

## Contract and composition

`src/dataset_batches.ores` exports `gather_supervised_batch_f64_checked_into(features, labels, indices, feature_scratch, label_scratch, feature_output, label_output)`. It joins two separately typed rank-2 `TensorF64` views into a consistent supervised minibatch using the same row selection order, including repeat sampling.

This is an algorithm in `.ores`, not a host/JS/Python implementation. It composes the `oreslang-tensor` pure `gather_rows_2d_f64_into` operation supplied by [tensor draft PR #5](https://github.com/ores-truffle-oreslang/oreslang-tensor/pull/5). The same `gather_rows_vjp_f64_checked_into` gives a mathematical reverse-mode rule for samples from a differentiable input tensor, correctly scatter-summing duplicated row gradients. **Random index generation or dataset membership does not become differentiable just because the selected tensor is.**

## Failure-atomic output semantics

The adapter validates upfront:
- both tensors are rank 2 and finite, with checked descriptor/storage extents and equal row counts;
- every selected index is within that shared row domain;
- both output and both scratch extents match the selected row count and respective feature/target widths;
- all four mutable destination buffers are mutually disjoint and none aliases either tensor input.

It gathers feature/label data into **separate caller-supplied scratch buffers**, then publishes both output buffers only after *both* pure row gathers succeed. A late failure may modify scratch but must not change either public output. Concurrent mutation of `List<float>` input storage is not proven safe; callers must enforce stability until compiler-recognized unique/frozen `Array<T>` supports verified immutable leases. Intentional sharing of the same *read-only input* backing between features and labels is not itself forbidden.

Zero-row and zero-feature data are admitted if the descriptors and output lengths match. The operation does not allocate output storage, perform I/O, shard across devices or use an actor per row.

## Worked training path

`examples/train_aligned_dataset.ores` defines four paired rows `x=[0,1,2,3]`, `y=[1,3,5,7]`, collects indices `[3,1,3]` into one minibatch, and invokes pure Oreslang `train_affine_step` (scalar MSE + SGD) on the selected values. Acceptance requires fitted `w≈2`, `b≈1`, loss less than `1e-6`, duplicate index order preserved, and invalid gather outputs unchanged. The fixture emits `ORES_ML_ALIGNED_DATASET_TRAINING_PASS` on completion.

This **is not** generic tensor autograd, model DSL compilation or actor pipeline training. The numerical VJP for row selection exists separately in the tensor module.

## Integration with actor pipelines

Data/ML actor stages retain the sole guest-owned `state { ... }` schema. Persist run/epoch IDs, bounded dataset cursor/partition positions, watermarks, live tensor lease/credit ledgers, gradient receipts, current parameter version and optimizer receipts **only within state**; no extra ordinary actor instance fields.

- `with` is a verified effect upper bound, never permission to access state by itself.
- `using state.read(|s| -> ...)` or sealed `self.handle_state(critical read |s| -> ...)` yields short read-only access.
- `using state.write(critical |s| -> ...)` or sealed `self.handle_state(critical write |s| -> ...)` yields exclusive state-write access through the actor's existing reader/writer gate.
- Stage performs data gather, tensor kernels and gradient VJPs **outside** live state loans. Pins must be independently validated immutable leases, not returned mutable state aliases.
- A processor cannot await downstream feedback while retaining state exclusivity. Message DATA/CONTROL identities and bounded byte/item admission are distinct from transport acknowledgement.
- Exactly-once local commit requires idempotency receipts; cross-process exactly-once external effects require an explicit durable sink/transaction protocol. Checkpoint replay and event-time watermark advancement require per-partition generation/sequence fencing.

## Verification and release path

`tests/dataset_batch_contract.test.mjs` is an independent host-only implementation of the mathematical contract, not compiled guest behavior. Isolated evaluation of its original fetched blob `f76e453312e1115fbc5ef2f2de38a43b305c909d` produced **7/7 passing tests, 3,374 assertions** including seeded aligned selections, output preservation on a simulated second-gather fault and loss-decreasing SGD.

`scripts/test-workspace.sh` explicitly requires an executable real `ORESLANG_BIN` and sibling `oreslang-autograd`/`oreslang-tensor` source. It invokes the new dataset-training example only when those prerequisites exist. An unconfigured runner exits 78 and **never** produces a success marker.

Next: pin dependency commits and run guest tests on Java/GraalVM JDK25 + Native Image and the supported LLVM subset; test compiler ownership and invalid-alias diagnostics; wire the consistent batch descriptor to bounded typed actor pipelines; then add deterministic resuming shuffled/windowed datasets, CPU multi-stage training, checkpoint/replay, and eventually GPU/distributed shards. No private Git trees should be mirrored to unrelated organizations without authorization.
