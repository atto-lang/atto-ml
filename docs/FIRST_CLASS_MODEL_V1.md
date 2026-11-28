# First-class `model` API — proposed user-facing contract

**Status: PROPOSED.** This document is the **ML specialization** of the authoritative common-source model contracts: merged model/autodiff proposal [#76](https://github.com/ores-truffle-oreslang/oreslang-common-source/pull/76) plus the general-purpose hardening draft [#93](https://github.com/ores-truffle-oreslang/oreslang-common-source/pull/93). `model` is not intrinsically an ML/neural-network type; the language-level abstraction is a nominal typed computational graph usable by other domains. This document is NOT an executable example on today's compilers.

## Preferred spelling

```oreslang
define model Classifier(
    features: int = 784,
    hidden: int = 512,
    classes: int = 10
) as
    Dense(features, hidden)
    ReLU()
    Dense(hidden, classes)
end

pub fnc train(dataset: Dataset) suspends {
    const trainer = ml.trainer(
        Classifier(),
        loss: CrossEntropy(),
        optimizer: AdamW(lr: 0.001),
        pipeline: ml.pipeline(
            stages: #auto,
            devices: #auto,
            schedule: #one_forward_one_backward
        ),
        microbatches: 16
    )

    await trainer.fit(dataset)
}
```

`Dataset` is a user-provided interface/pseudotype here; it is not yet a stable exported type. `Dense` should map to the same matrix primitive as existing/planned `Linear`. The v1 numerical reference stack is **f64 on CPU**, pending generic dtype/contiguous `Tensor<T>` backing.

## Why this API, rather than separate `model = ...` and `pipeline = ...`

- The `define model` declaration defines an immutable, nominal, type-checked ordered computation graph. Its constructor creates a *new* model value/instance. ML trainable ops introduce independently owned parameter storage; a stateless non-ML or inference model may have no mutable parameter state at all.
- `ml.trainer` owns the training session and optimizer state. Explicit `pipeline:` selects an execution/scheduling strategy; there is no need to materialize an intermediate user variable for the same plan.
- `model` is NOT an actor and is NOT an alias for `ActorStream`. Runtime stages may be one-to-many groupings of graph operations. Model ownership remains explicit even when stages are repartitioned.
- Automatic gradient graph generation lives in `oreslang-autograd` and future compiler/actor lowering. A `backward` handler does not require user code for built-in layers. A custom operation needs a validated VJP **only when an explicit requested derivative path traverses it**; forward-only use of a nondifferentiable op remains legal.
- Neither `|>` (function pipeline) nor `pipe` (actor pipeline) becomes an implicit differentiation operator.

## Small CPU version (planned defaults)

```oreslang
pub fnc train(dataset: Dataset) suspends {
    const trainer = ml.trainer(
        Classifier(),
        loss: CrossEntropy(),
        SGD(lr: 0.01)
    )
    await trainer.fit(dataset)
}
```

The shorthand MUST be equivalent to a one-stage, CPU, fill-and-drain, one-microbatch plan until a higher-performance policy is explicitly chosen. No silent GPU use, remote actor creation or unconstrained memory usage.

## Inference and explicit actor scheduling (future API)

```oreslang
const instance = Classifier()
// Proposed ML alias: inference returns a typed tensor, no optimizer or gradient tape.
// The general-purpose model contract uses execution/run semantics; "predict" is ML vocabulary.
const logits = instance.predict(inputs)

// Proposed optional explicit graph placement.
const plan = ml.pipeline(
    stages: 3,
    devices: #auto,
    schedule: #one_forward_one_backward
)
```

The compiler may fuse operations or shard expensive tensor kernels; a logical `Dense` or `ReLU` call need not spawn an actor. Runtime forwards activations in graph order and propagates VJP gradients backward along dependencies, with repeated-edge/fan-in accumulation.

## Production invariants

1. Every training microbatch is keyed by run/step/microbatch/graph-generation/weight-version; forward and backward must pair to the same version.
2. Activation/gradient buffers are actor-owned or validated immutable/transfer leases. Borrowed mutable buffers never escape across a suspending forward/backward operation.
3. Each downstream graph edge contributes at most once; accumulated gradients complete before the single optimizer step commit. Cancel/retry/restart cannot duplicate updates.
4. Bounded mailboxes plus explicit backpressure protect memory; a serial stage returns after publishing forward data instead of awaiting its own backward reply.
5. CPU deterministic convergence, independent finite difference oracles, cancellation/replay negative cases, and real Java/LLVM compiled guest executions are release gates.

## Relationship to the general-purpose model layer

This package does not own the meaning of `define model`. Common-source/compiler semantics own nominal model identity, canonical static configuration, typed graph ports, effect/resource admission, graph generation and model-instance ownership. `oreslang-ml` contributes ML operation signatures, parameter schemas, losses, optimizers, training sessions and optional differentiation/training execution.

Consequences:

- non-ML model declarations must compile without importing this package;
- Tensor/Autograd are not mandatory dependencies of `core::model`;
- `Dense`/`ReLU` are ordinary admitted model operations in the ML domain rather than privileged language syntax;
- freestanding/no-runtime-allocation/full-AOT eligibility is derived from the chosen operations/backend and does not imply hard real time;
- actor/GPU/distributed placement remains an execution-plan concern, not part of model type identity.

## Delivery boundaries

The existing [P0 ML issue #2](https://github.com/ores-truffle-oreslang/oreslang-ml/issues/2) and its draft PRs remain focused on pure `.ores` scalar/tensor reference training. This guide does **not** supersede them.

Deliver first: pure `Linear/ReLU/Sequential/MSELoss/SGD` and tensor-valued VJPs on owned contiguous storage. Then implement `define model` in Java and LLVM as compiler sugar lowering to those verified library graph primitives. Introduce actor-stage scheduling, 1F1B, acceleration and distributed execution behind independent evidence gates. No claims of complete syntax, native compiler parity, autotuned scheduling or production training in this document.
