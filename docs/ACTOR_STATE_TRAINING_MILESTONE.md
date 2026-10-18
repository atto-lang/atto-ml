# Actor-owned ML stages: sealed state + reader/writer admission milestones

**Status 2026-10-18: source fixture and host oracle added, NOT yet compiled and run on its pinned exact Java or LLVM compiler source.** This is a stacked draft on ML consumer API PR #5 and complements common-source autograd draft #76 plus actor-state/MVCC draft #78. Do not merge or advertise production training on metadata-only evidence.

## Language semantics (mandatory, not a new neural-network API)

A concrete Oreslang actor has at most one effective guest-owned `state { ... }` schema. All persistent actor weights, versions, optimizer slots, activations and gradient receipts must be transitively owned from this **single state variable**. Ordinary persistent actor fields, a second state declaration and a user-side storage map keyed by actor ID are forbidden. Runtime-owned scheduler tickets and mailbox metadata remain separate and inaccessible to guest code.

- **`with state.read`** on a function/handler is the declared upper bound permitting *read effects*, not a lock or capability.
- **`with state.write`** is an upper bound permitting *write effects*. Expressions that both read and write the same field require both effects; write-only assignments must not cause speculative reads.
- **`using state.read(|s| -> ...)`** grants a read-only ephemeral loan on the actor's one state domain; `self.handle_state(critical read |s| -> ...)` is the canonical sealed intrinsic equivalent.
- **`using state.write(critical |s| -> ...)`** and `self.handle_state(critical write |s| -> ...)` grant *exclusive* ephemeral write admission. Write access cannot be inferred from `with`, `serial`, `pub`, actor kind, or the caller's mutable reference.
- There is **one reader/writer gate per actor state domain**. Read cohorts may share admission only where the runtime also authorizes concurrent execution. Writer admission is exclusive, FIFO writer-aware, cancellable and fenced by actor incarnation.
- A callback cannot retain/capture/return a state loan, borrow reference-bearing mutable fields outside an ownership-proof boundary, or suspend while the loan remains live. No implicit conversion between state-owned tensor buffers and sendable handles.

The state RW gate is **not** the same as the actor's native execution-thread limit and does not replace single-owner tensor storage. For maxThreads=3 actors, three physical executing turns are admissible only after the compiler, scheduler, and state gateway independently enforce loan exclusivity and proper lifetime fencing. Default (1,1) execution remains valid.

## New scalar source fixture

`examples/train_actor_state_scalar.ores` is a deliberately small `int` affine stage:

- Sole `state { weight, version, accumulated_gradient, pending_gradients, step, seen_first, seen_second }`.
- The current scalar actor fixture performs checked integer forward multiplication **inside** a short exclusive `using state.write(critical ...)` callback when recording the forward context. Moving computation outside the loan requires a future immutable, version-pinned snapshot API; this source is not evidence that such a lease is implemented.
- The scalar VJP currently multiplies the *saved forward input* by the checked upstream integer within an exclusive state callback, then records a deduplicated receipt. Bound checks precede receipt mutation. Production tensor VJPs must compute outside loans on immutable parameter snapshots and revalidate their envelopes during a later critical receipt transaction.
- An incomplete optimizer step returns a deliberate rejection rather than publishing a partial update. The synchronous step compares version and step inside `self.handle_state(critical write ...)` before updating weight and version together.
- `self.handle_state(critical read ...)` observes the committed version. An old-version gradient is rejected after commit.
- The actor is contacted only through inbound messages. No `ActorRef.someMethod()` public RPC.

The fixture is a **scalar control/gradient bootstrap**, not autograd tape recording, a tensor actor kernel, a float/f64 optimizer, multi-stage backprop, or a crash-safe distributed transaction. Its fixed two-microbatch IDs and integer update may be replaced by dynamically bounded keyed ledgers and immutable tensor version leases in production. The source assumes arithmetic remains in range; production must add verified checked tensor arithmetic before an optimizer commit.

Expected stdout (one value per line):

```text
-6
-8
20
-7
30
-8
2
-4
-2
5
5
2
-1
50000
50000
100000000
200000000
-8
2
3
-1
10
10
```

`tests/actor_state_scalar_contract.test.mjs` provides **host-only source-structure checks and an independent numerical scalar-VJP oracle**, not a substitute for compiling the source.

`scripts/verify-actor-state-guest.sh COMMAND [ARGS...]` is an exact-output checker that invokes the supplied *already installed* Oreslang runner and appends the fixture path. It fails on compiler failure or stdout mismatch; it does not select/build/download the compiler and is not run by the host-only CI workflow.

## Versioned tensor-stage end state

The mature stage remains actor-state-only:

```oreslang
define actor DenseStage as
    state {
        // All of these library types are PROPOSED and not compiler-ready.
        weights: FrozenParameterVersions;
        active_version: int = 1;
        optimizer: OptimizerState;
        contexts: ActivationLedger;
        gradients: GradientLedger;
        commit_receipts: CommitLedger;
    }
    // Forward receives a versioned envelope, pins immutable storage under a
    // short write loan, computes outside the loan, and self.forward()s.
    // Backward validates exactly-once (run, graph, step, microbatch, edge,
    // actor epoch, weight_version), executes the VJP outside the loan,
    // then records/merges contributions under a write loan.
end
```

Every pinned weight version is an independently validated immutable storage lease, not a leaked `state.weights` borrowed pointer. Parameter buffers may be allocator/device-owned with **handles and ownership recorded inside state**; no user-owned second actor field is introduced. The gate protects atomic *metadata and publication* transitions, while frozen storage plus pinned lifetimes protect long-running CPU/GPU computation.

Prepare candidate parameters and optimizer slots independently **outside** the write loan, then atomically replace both under one critical commit with a compare-and-swap check. Failed prepare, cancelled step and stale-generation gradients must not alter committed weights. Each cross-actor shard has its own gate; a distributed coordinator additionally needs a durable generation/commit protocol and consistent inference version selection.

## Milestone gates (none may be inferred from source fixtures alone)

| Gate | Deliverable | Acceptance |
| --- | --- | --- |
| M1 | Scalar source actor on Java/Graal | Exact Java source SHA; actual guest compilation/execution; pinned stdout, negative compiler fixtures and stopped-actor retirement |
| M2 | LLVM scalar parity | Same source where representable, otherwise explicit supported-subset paired fixture, signed source/IR exact-tree evidence and UBSan/ASan |
| M3 | State RW execution | Multiple reader permits, writer fairness, cancelled granted/queued permits, no read->write upgrade, 1/3 execution-thread configurations, no loan across suspension |
| M4 | Owned tensor weights | Contiguous f64 CPU storage; generation/epoch-bound read-only weight leases; finite-difference validated VJPs; memory and lifetime bounds |
| M5 | Actor pipeline autograd | Two/three actual stages with forward/backward fan-in, bounded microbatches, 1F1B, duplicate-edge and stale-version rejection, one optimizer update/step |
| M6 | GPU & distributed | Device event lifetime, NUMA placement, shard reductions, durable commit fencing and restart/cancel fault injection |

**Current evidence:** Isolated V8 tests of source spelling/numerical oracle, and separate common-source host-only MVCC/RW admission simulators; no verified exact-head full Java Maven, Native Image, LLVM native guest, or multi-stage ML training run for this new branch. CI failures due to inaccessible runner budget are not passes.

### Pre-1.0 rejection matrix for compiled tests

Reject: ordinary actor `val Linear layer` field, duplicate `state`, direct `self.state`, state mutation under a read loan, write loan without `critical`, `with` underrun, signature-level `using`, escaping references through lambda return or captured closures, suspension with live state loan, duplicate gradient edge/microbatch, unrecognized graph/actor generation, stale weight version, and commit with incomplete fan-in.

Future compiler acceptance must check both Java and LLVM source diagnostics, not rely on host regex scans. Private Git trees should remain inside explicitly authorized boundaries; no cross-org mirroring while the source-exposure incident remains unresolved.

## Canonical actor lifecycle and actual guest-verification boundary (October 18)

The scalar actor fixture now uses `await worker.get_ready()` before its first `worker.send(...)` and `await worker.get_done()` after the terminal message. Older `await worker.ready` / `await worker.done` spellings do not match the current Oreslang ActorRef lifecycle contract and are prohibited in this fixture. The byte-identical guest fixture in `oreslang-source.java` PR #1347 was synchronized, and `MlActorStateScalarGuestTest` pins its Git blob `9171c320420a128ca61cc193c9dd4db985222552`. The Java test also executes the parser, actor-state contract, declared effects, type checking, `OwnershipChecker`, and a real Graal polyglot context **when a permitted runner becomes available**.

Source-contract tests for the scalar fixture now require readiness before the first message and done only after all messages; **10/10 isolated host V8 tests passed**, against source blob `9171c320420a128ca61cc193c9dd4db985222552` and test blob `f1b0ce19960390c8d06677106b9a0af578ce5978`. This only verifies the source and an independent numerical model, **not** that the Oreslang scheduler executes the barriers correctly.

A targeted read-only same-org Java/Graal Actions workflow `.github/workflows/ml-scalar-actor-guest.yml` was added on the Java PR branch. It verifies the exact checkout revision and invokes the real `mvn -B -ntp -Dores.runtime.carriers=native -Dtest=MlActorStateScalarGuestTest test`, then full `verify` only after success. Actions run [#39161272389](https://github.com/ores-truffle-oreslang/oreslang-source.java/actions/runs/39161272389) reports `completed/failure` with **no job steps**; it cannot be counted as JUnit test execution or evidence of actual runner minutes exhaustion.

Separately, an independent source-free synthetic lifecycle model validates readiness, drain-before-done, duplicate mail, bounded admission, cancellation and incarnation fencing. It is not a Truffle scheduler or a compiler test. Cross-org private-source copying remains prohibited by open security incidents #22 and #77 absent independently recorded approvals.

## Audit: bounded integer gradients and fail-atomic receipt handling

The scalar guest example used unchecked integer multiplication/accumulation and modified receipt state before arithmetic. Those operations were not safe if incoming messages supplied arbitrary `x` or `upstream` values. Now:

- Forward validates `x` and `weight` within `[-10000,10000]` **before** setting `issued_first/second`. Product magnitude is bounded by `100000000`.
- Backward validates `upstream` within the same domain **before** setting either `seen` flag or changing the sum. Its input comes from the previous bounded forward context, not from the backward message. With at most two receipts, total gradient magnitude is at most `200000000`, safe for signed 32-bit intermediates.
- Commit validates complete receipt fan-in, then rejects version/step counters at or above `1000000` and a proposed weight outside `[-10000,10000]`, **before any weight, counter, or receipt mutation**. The largest candidate intermediate magnitude is `200010000`, also within signed 32-bit range.
- An invalid incoming numeric value and an oversized optimizer candidate return the scalar demonstration's `-8` marker; exhausted counters return `-9`. Rejected commits leave receipts and the current version/weight intact, with no silent wraparound or forged progress.

The updated guest output exercises malformed initial forward/backward inputs, valid retry, high but bounded per-microbatch gradients, late out-of-range optimizer rejection, and an unchanged weight/version after that rejection. Java's pinned copy and JUnit expected output use source blob `5a6c8240ea257b0cee14f0d086ab2a14f0b369d3`. The 20-line golden lives in `tests/golden/actor_state_scalar.stdout`.

**Important API limitation:** negative integer status codes are only illustrative. Legitimate computed gradients or predictions can also be negative, so they are not a collision-free production error channel. A production training actor must use a typed `Result` or sum/enum outcome and messages with separate success and error variants. A failed optimizer candidate intentionally remains pending; the application needs a separately authorized cancel/reset/retry protocol instead of silently discarding gradient receipts. This reference does not prove arbitrary-precision arithmetic, LLVM execution, distributed transactional publication, or fully isolated tensor buffers.

The new `tests/actor_state_bounded_numeric_contract.test.mjs` preserves five independent host-only regression cases, including 4096 seeded gradient-bound trials. Together with the updated actor source contract, **17/17 isolated V8 source/numerical cases passed** against current blobs `5a6c8240ea257b0cee14f0d086ab2a14f0b369d3`, `2c533e99b9cb60ab8c859cdb84968f763e6d6005`, and `71efc113c4de7d045a30ceaeea69e34267139a52`. The targeted Java/Graal guest remains **NOT VERIFIED EXECUTED**. No cross-organization private source transfer was made.


## Typed scalar API and rejected-step compensation (2026-10-18)

`src/training/scalar_checked_result.ores` now defines a *pure*, caller-owned CPU arithmetic reference with `Result<int, String>` returns for forward, gradient staging, commit candidate, and rejected-step retirement. This avoids interpreting legitimate negative predictions, such as `Ok(-8)`, as integer error statuses. Its string labels are deliberately **not** claimed to be a final nominal `TrainingError` enum; an exhaustive checked enum/sum type is the next compiler-compatibility gate.

`examples/typed_scalar_step_result.ores` is an actual Oreslang guest acceptance target for those Result operations; `scripts/verify-typed-scalar-guest.sh` requires a clean ML Git checkout at a caller-provided exact revision, absolute compiler executable hash, and exact success marker. **No compiled guest run has passed.** Host-only `tests/scalar_checked_result_contract.test.mjs` covers four independent cases, including 4096 seeded integer-domain trials; local isolated execution passed 4/4.

The existing stateful scalar actor fixture now has a **typed `Result<int, String>` compensating method**, `cancel_rejected(expected_step, expected_version)`. It only retires a step after both forward contexts and gradient receipts are present and the optimizer candidate has actually failed its bounded-range guard. It does not publish a weight or optimizer update. Within a short exclusive state write loan, it clears pending microbatch contexts/receipts and advances both step and version together; late gradients for the failed step are rejected by the version fence. Valid candidates **cannot** be discarded through this method and must follow the existing commit path. Production control envelopes additionally require authenticated/authorized caller intent, graph/run/incarnation identity, and explicit retry/cancel policies; the fixed-message test fixture is not an authorization design.

The scalar actor guest golden now has **23 lines**, including rejected optimizer candidate, successful compensating version retirement, stale-message rejection, and a fresh post-retirement forward. Java/Graal PR #1347's test fixture and Git-blob pin were synchronized exactly. The actor's original forward/backward/commit operations still return signed integer status sentinels, so **the P1 typed migration is partial**; it must not be described as full typed-error parity until those methods switch to collision-free results. Same-org Java Actions remain blocked before recorded job steps; numerical host references are not runtime evidence.
