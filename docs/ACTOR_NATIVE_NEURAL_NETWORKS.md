# Actor-native neural networks: implementation and release gates

Status: **prototype / source-only**, not an independently executed Oreslang
compiler receipt. This document is an implementation contract, not a statement
of five-backend parity.

## Ownership and factoring

- `oreslang-data-science`: nullable datasets, columnar ingestion, windowed
  minibatches and data-oriented CPU preprocessing.
- `oreslang-tensor`: a single owned `Array<T>`/NdArray storage authority,
  checked shape/stride/broadcast/conv/matmul kernels. Current `List<float>`
  reference is temporary; compiler loans and safe zero-copy views are missing.
- `oreslang-autograd`: tensor reverse-mode VJPs, immutable forward tapes,
  graph generation leases, fan-in and deterministic accumulation.
- `oreslang-ml`: pure `.ores` layers, `Sequential` / `ModelGraph`,
  SGD/AdamW, native actor-based training, inference and serving.
- `oreslang-ml-conformance`: independent oracles, finite-difference checks,
  artifact provenance, backends, negative tests, real device receipts.

No second tensor storage runtime or hand-written per-model backward engine.
All public Oreslang algorithm implementations stay in `.ores`.
Rust is acceptable as an **independent test oracle**, never as an ML runtime.

## Stage 1 — eager tensor training

Current integration target: `examples/train_tensor_mlp_vjp_cpu.ores` on this
branch, stacked on `oreslang-tensor` P0 + `oreslang-autograd` PR #10.
For each training sample:

1. Forward: X @ W1 + b1 -> ReLU -> H @ W2 + b2 -> MSE.
2. Backward: MSE VJP -> matmul dH,dW2 -> saved-ReLU VJP -> matmul dX,dW1.
3. Accumulate deterministic minibatch gradients, preflight finite candidates.
4. Publish parameters after preflight. The standalone sample is not yet an
   actor-owned atomic publication; the production trainer must fence it.

Single-process CPU reference first, then full tensor `Module`/Optimizer
abstractions. A two-layer 1->2->1 MLP is not yet a general-dimension MLP or
end-to-end generalized reverse-mode graph scheduler.

`src/adamw.ores` is a bounded pure-Oreslang reference with bias correction,
verified disjoint scratch buffers, and staged publication of parameters and
first/second moments. Require separate optimizer state owner per model instance
and a versioned critical-write commit in the actor trainer. No step update may
publish only a subset of its tensors.

## Stage 2 — actor graph (each actor is long-lived, not each scalar FLOP)

**NeuronActor demo:** demonstrate 4-16 long-lived stateful neurons with
mailbox-driven activations and local state in the AI coordination demo.
This is for semantics, visualization and supervised lifecycle tests, not a
claim of throughput comparable to tensor kernels.

**Production:** actor per layer, pipeline stage, or data/tensor shard. No actor
per tensor element. Roles:

- `TrainingCoordinatorActor`: deterministic run/graph identity, dataset and
  step schedule, microbatch credits, cancellation, supervision and checkpoints.
- `LayerActor`: owned parameters and immutable weight snapshots, saved
  activations tagged by generation, forward/backward worker scheduling.
- `GradientReducerActor`: accept exactly one gradient contribution per
  (run, incarnation, graph generation, step, weight version, microbatch, edge),
  bounded fan-in and deterministic ordered reduction.
- `OptimizerActor`: sole writer, SGD/AdamW states, candidate preflight and
  **all-parameter** CAS publication under one exclusive state loan.

Message protocol must be *typed*, not `ActorMail<String>` string switches.
Sealed actor state is reachable only via compiler-authorized
`handle_state(critical-read/critical-write)` loans. No `await` or actor send
under an exclusive loan. Each message carries a runtime-authenticated actor
principal/capability, immutable run and incarnation IDs, graph generation,
step, weight version, microbatch ID and stable edge ID. Integers supplied by
an untrusted caller are **not** authentication.

Enforce mailbox admission quotas and network/CPU/GPU budgets; supervisor
CONTROL messages are priority path, a panic is never interceptable by the
panicking actor. After restart, stale in-flight gradients are rejected by
incarnation/version fencing. An unrecoverable unknown optimizer publication
outcome must be reconciled against a durable idempotency-key ledger, not
retried blindly.

First valid schedule: bounded `fill_drain`; subsequently prove 1F1B
pipelining with generation-safe activations. Weight snapshots must remain
frozen across the entire gradient accumulation window. Determinism needs
stable edge ordering, rounding/dtype policy and exact seed specification.

## Stage 3 — neural families (same autograd and storage)

| Family | First operators | VJP requirements | Acceptance |
| --- | --- | --- | --- |
| CNN | 1D/2D conv, pooling, padding/stride | input/filter/bias, overlap summation | finite differences, tiny image classification |
| Attention | stable softmax, Q/K/V matmuls, masking | softmax Jacobian-vector product, dropout seed rules | causal mask, row-stable invariance, reference attention |
| Transformer | embeddings, layernorm/RMSNorm, attention, FFN | shared tensor graph, residual fan-in, cross-entropy | tiny next-token model learns a bounded corpus |
| KAN | cubic B-spline and packed KANLayer | coefficient/input/bias VJPs | build on ML draft PRs #14-#19, independent golden data |
| Distributed | tensor sharding, reduce, pipeline | duplicate-safe deterministic fan-in | single-process vs split actor equality |

**No hidden CPU fallback** on an explicitly requested GPU target. GPU
execution needs real device and cancellation/transfer receipts, not simulator
success. CNN, attention, transformer and KAN are distinct acceptance steps,
not complete simply because this architecture document exists.

## Stage 4 — proof and promotion

For **each** concrete fixture, record:
- Exact producer Git blob/tree/commit SHAs, compiler binary/source SHA,
  dependency lock, toolchain, invocation, exit code, stdout and stderr.
- True executed job steps. A zero-step Actions run is `NO_STEPS`, not a
  language or compiler failure and never a success.
- Independent analytic and central finite-difference VJP oracles, branched
  DAGs/repeated edges, zero dimensions, mixed broadcasts, stale leases,
  duplicate deliveries, nonfinite/overflow/shape violations.
- Java/GraalVM and LLVM **real guest** acceptance first. Then separate
  WASM, Forge, .NET and Cranelift capability reports; unsupported is not passed.
- ASan/UBSan where applicable; hard/soft realtime budgets reported separately.
- One explicit release matrix: `REFERENCE_ONLY`, `NO_STEPS`,
  `EXECUTED_UNATTESTED`, `VERIFIED_BACKEND`. No inferred parity.

Existing tracking: ML #2 and #6, Tensor #2, Autograd #2,
ML Conformance #4 and #6. Source grammar/ownership work must be integrated
before sealed-state actor programs are called compiler-verified.
