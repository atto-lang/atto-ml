# Columnar data → tensor → autograd → CPU training integration

**Status October 18, 2026: executable-intended Oreslang source and independent host-only reference tests committed. Real Java/GraalVM or LLVM guest execution is NOT VERIFIED.**

## Implemented in the source fixture

`examples/train_columnar_tensor_autograd_cpu.ores` uses existing public Oreslang functions, not a new numerical framework:

1. `oreslang-data-science`: `compact_supervised_f64_into` intersects nullable feature and target masks, retaining matching original row IDs; fixed data deliberately excludes two source rows.
2. `oreslang-tensor`: `gather_rows_2d_f64_into` resamples row 3 twice and carries matching labels. `matmul_2d_f64_into` applies a checked [1,1] identity projection to the [5,1] gathered tensor.
3. `oreslang-autograd`: `TapeF64`, `forward_checked_into` and `backward_checked_into` execute a full two-affine-layer ReLU squared-loss graph, with gradients accumulated over all five examples before each SGD update.
4. `oreslang-autograd` + `oreslang-tensor`: `row_gather_vjp_with_saved_f64_checked_into` verifies the saved selected source values and uses duplicate-aware scatter-add to propagate scalar input gradients back to the source tensor.

This is a **CPU numerical bridge**. Matmul is deliberately an identity: its analytical VJP is identity, but general matmul reverse-mode rules, generic tensor tapes, actor-resident learned parameter storage, GPU kernels, and distributed optimization are **not** implemented here.

The program trains `x → affine1 → ReLU → affine2 → MSE` using five resampled training rows for 1,000 SGD iterations. Its explicit guest acceptance marker is:

```
ORES_ML_COLUMNAR_TENSOR_AUTOGRAD_CPU_PASS
```

The numerical expectation was independently verified by a 13-node reverse-mode host reference. The host oracle's ten cases cover: mask intersection and row provenance, mismatched mask/error handling, identity matmul bounds, DAG validity, parameter and input finite differences, loss convergence, saved row-gather repeated-index gradient summation, and 256 seeded linear adjoint tests.

## Fail-closed real compiler gate

Set up a workspace with *sibling* repositories `oreslang-ml`, `oreslang-data-science`, `oreslang-tensor`, `oreslang-autograd`. Run:

```sh
ORESLANG_BIN=/absolute/path/to/pinned/oreslang \
  sh oreslang-ml/scripts/test-columnar-cpu-integration.sh
```

The verifier refuses an absent/non-executable/nonabsolute compiler, missing dependencies and SHA-inconsistent library blobs. All eight direct dependency source files are pinned by Git blob digest so a different branch version cannot silently give success. The verifier invokes the real CLI, compares stdout against the committed golden, fails on nonzero exit/output mismatch and records the guest fixture blob. This is **file-blob** provenance, not an assertion that four entire repository trees are byte-for-byte identical. Record exact compiler source/tree revision and logs separately.

No private source is copied across GitHub organizations. The open security incidents in platform-conformance #22 and #77 prohibit cross-org private-source mirrors until independent review/authorization. Funded *source-free* numerical CI is valuable but not a substitute for native compiler execution.

## State, effects and actor pipeline boundary

This standalone numerical example is intentionally **not** an actor implementation and does not access `state`. Future trainer actors MUST own all guest-persistent weights, optimizer state, in-flight forward contexts and gradient receipts through a **single** `state { ... }` declaration. `with state.read` / `with state.write` declare permitted effects; `using state.read`, `using state.write(critical ...)` and sealed `handle_state(critical read/write ...)` acquire checked lexical loans from the actor's reader/writer lock. Nothing may retain a state loan across `await`, GPU work, transport, or a suspending callback.

Before moving this example into actor messages, production must provide immutable **leased** parameter snapshots and activation storage, a bounded microbatch ledger, checked gradient aggregation, and a short compare-and-publish optimizer transaction guarded by (run, epoch, graph, step, microbatch, edge, version). The current `TensorF64` and `DenseSupervisedBatchF64` wrap caller-mutable `List<float>` storage; `pub val` is **not** deep immutability, and those buffers cannot be sent to another actor as immutable state without compiler-verified ownership transfer or explicit frozen copy.

## Next acceptance milestones

| Level | Acceptance evidence |
| --- | --- |
| CPU-0 | Source/host reference, numeric convergence and algebraic VJP checks (**host-only PASS 10/10**) |
| CPU-1 | Actual Java/GraalVM guest run of pinned source with golden stdout; recorded parsed effect and ownership checks (**NOT_EXECUTED**) |
| CPU-2 | Full tensor matmul/affine VJPs, automatic saved tensor graph, independently frozen output buffers; finite differences |
| ACTOR-1 | Ingress/forward/backward/optimizer actors, only one `state` each, `with` effect and `using`/RW gate, no loan escapes; two concurrent microbatches |
| ACTOR-2 | Versioned snapshot pin/release, cancellation/de-duplication, full fan-in gate, fault replay and atomic optimizer commits |
| LLVM | Same semantically supported guest source, native builds and sanitizers on exact-tree verified commit |
| GPU | Explicit device lifetime, memory limits, async events, distribution/restart and performance benchmarks |

**Do not mark this as AI/ML pipeline parity solely because a host numerical oracle or a draft workflow passed.**
