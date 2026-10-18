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
- Invalid inputs, gradients, incomplete fan-in, unsafe candidates and counter exhaustion now yield typed `Err(String)` values, not integer status sentinels. Rejected commits leave receipts and current weight/version intact, with no silent wraparound or forged progress.

The current 24-line golden exercises typed rejection markers, valid retry, two bounded large gradients, late optimizer rejection, compensating cancellation, stale step rejection, and a valid negative `-10` forward. Java PR #1347's guest fixture and golden are byte-identical to the ML copy.

**Current API limitation:** all four scalar actor stateful routines now return `Result<int, String>`, so valid signed values cannot collide with failures; the error field remains a provisional *string* rather than a nominal `TrainingError` enum. Failed optimizer candidates remain pending until an explicit cancellation/retirement request. The fixed guest messages do not establish authenticated cancellation authority. This reference does not prove arbitrary-precision arithmetic, LLVM execution, distributed transactional publication, or fully isolated tensor buffers.

The new `tests/actor_state_bounded_numeric_contract.test.mjs` preserves five independent host-only regression cases, including 4096 seeded gradient-bound trials. Together with the updated actor source contract, **17/17 isolated V8 source/numerical cases passed** against current blobs `5a6c8240ea257b0cee14f0d086ab2a14f0b369d3`, `2c533e99b9cb60ab8c859cdb84968f763e6d6005`, and `71efc113c4de7d045a30ceaeea69e34267139a52`. The targeted Java/Graal guest remains **NOT VERIFIED EXECUTED**. No cross-organization private source transfer was made.


## Typed scalar API and rejected-step compensation (2026-10-18)

`src/training/scalar_checked_result.ores` now defines a *pure*, caller-owned CPU arithmetic reference with `Result<int, String>` returns for forward, gradient staging, commit candidate, and rejected-step retirement. This avoids interpreting legitimate negative predictions, such as `Ok(-8)`, as integer error statuses. Its string labels are deliberately **not** claimed to be a final nominal `TrainingError` enum; an exhaustive checked enum/sum type is the next compiler-compatibility gate.

`examples/typed_scalar_step_result.ores` is an actual Oreslang guest acceptance target for those Result operations; `scripts/verify-typed-scalar-guest.sh` requires a clean ML Git checkout at a caller-provided exact revision, absolute compiler executable hash, and exact success marker. **No compiled guest run has passed.** Host-only `tests/scalar_checked_result_contract.test.mjs` covers four independent cases, including 4096 seeded integer-domain trials; local isolated execution passed 4/4.

The existing stateful scalar actor fixture now has a **typed `Result<int, String>` compensating method**, `cancel_rejected(expected_step, expected_version)`. It only retires a step after both forward contexts and gradient receipts are present and the optimizer candidate has actually failed its bounded-range guard. It does not publish a weight or optimizer update. Within a short exclusive state write loan, it clears pending microbatch contexts/receipts and advances both step and version together; late gradients for the failed step are rejected by the version fence. Valid candidates **cannot** be discarded through this method and must follow the existing commit path. Production control envelopes additionally require authenticated/authorized caller intent, graph/run/incarnation identity, and explicit retry/cancel policies; the fixed-message test fixture is not an authorization design.

The scalar actor guest golden now has **24 lines**, with separate `REJECTED` markers and a valid `-10` result. Java/Graal PR #1347's fixture and golden remain exactly synchronized. **The three original actor methods have now been migrated to `Result<int, String>`**, completing typed value/error separation, while nominal exhaustive `TrainingError`, authenticated cancellation and backend execution remain outstanding. Same-org Java Actions remain blocked before recorded job steps; host numerical references are not guest execution evidence.


## Audit continuation: accumulator provenance and compiler-test hygiene

`checked_scalar_gradient` now checks the **receipt-count/accumulator feasibility invariant** before returning an updated `Result`: zero receipts require a zero accumulated gradient, and a single receipt cannot contain a gradient whose absolute magnitude exceeds 100,000,000. This closes an API-level path for constructing numerically bounded but logically impossible staged-gradient values. The two-receipt candidate still requires validation and publication under the actor's own exclusive state admission. These checks are *not* proof of cryptographic receipt authenticity, run identity, or complete borrow/lifetime safety.

The real guest target `examples/typed_scalar_step_result.ores` now tests two impossible accumulator combinations and a boundary-legal gradient. Six isolated typed source/numeric tests, twelve additional scalar actor tests and five bounded-arithmetic tests passed in the retrieved in-process V8 reference harness: **25/25 total**. No real guest compiler was run.

In the Java source-owned PR #1347, the `MlActorStateScalarGuestTest` suite previously referenced `invalidSource` inside its positive-control method, where it was undefined. That compile-breaking reference was fixed. Each negative case now checks that `Parser.parse(invalidSource)` succeeds, then checks that explicit contract/effect/ownership admission rejects it, before asserting that real `context.eval` fails without executing `main`. Three isolated host-only structural checks pass, but JUnit compilation itself remains unverified due to zero-step source-org Actions failures.

The funded *independent source-free* reference carrier passed **65/65 actual Node tests** on [run 39164887880](https://github.com/sagitta-stack/sagitta-infra/actions/runs/39164887880), exact carrier head `4f596cea2e160f0ebe4ebe8067dd57893739f043`. Its admission scanner verified no tracked native-source file extensions or embedded native declarations within its independent `.mjs` suites. This is a guardrail, not a full content-similarity forensic proof. Logs explicitly report `PRIVATE_ORESLANG_SOURCE=NOT_TRANSFERRED` and `ORESLANG_GUEST_EXECUTION=NOT_EXECUTED`. The private-source export restrictions of incidents #22 and #77 continue to apply.

## Typed actor state-machine milestone — latest audit and exact tests

- The four actor operations `eval_affine`, `accumulate`, `commit`, and `cancel_rejected` now all return `Result<int,String>` under the actor's existing single `state {}` schema and scoped `using state.write(critical ...)` / `handle_state` gates. The CLI's `print_outcome` prints `REJECTED` for errors, otherwise unwraps and prints a valid signed numerical result.
- The input/gradient admission sequence and candidate checks remain **before** receipt or committed-weight mutations. A rejected candidate can be retired only after complete fan-in and only through the explicit compensation path. Step and version advance together on commit or compensation. Full nominal error typing and caller authorization are **not** yet implemented.
- The actual `.ores` guest and 24-line stdout golden in `oreslang-ml` use Git blobs `26a3cb34b126185d56f5d2c7ff370cab953b0b4b` and `867f4ca71a20b58b1f35e239d4bef160f7ef5342`, identical to Java PR #1347's source and golden fixtures, respectively.
- Isolated host-side source/numerical tests: 27/27 previous checks and 8/8 new transaction-state checks, **35/35 total**. The latter include 4096 seeded candidate schedules, negative success, duplicate/orphan/stale rejection, complete fan-in, safe cancellation and exhausted counter boundaries. These are not a compiler execution receipt.
- Independently authored **source-free** funded GitHub Actions [run #39166122090](https://github.com/sagitta-stack/sagitta-infra/actions/runs/39166122090) executed **78/78 Node tests** at exact carrier Git head `f828000467436529016a96c8f1c2cc3899cef2df`. The carrier's independent scanner passed and log states `PRIVATE_ORESLANG_SOURCE=NOT_TRANSFERRED`, `ORESLANG_GUEST_EXECUTION=NOT_EXECUTED`, `EXACT_SOURCE_TREE=NOT_MIRRORED`. **Do not** infer Oreslang compiler parity from these synthetic checks.
- The Java/GraalVM source-org test still fails before any recorded runner steps. Private-source transfer incidents #22 and #77 remain open, so no private tree was copied across organizations.


## Run/incarnation fencing and internal compensation (October 18 follow-up)

The scalar guest now stores **demo-only** `run_id=101` and `incarnation=1` inside its sole actor `state { ... }`. All four operations (`eval_affine`, `accumulate`, `commit`, `cancel_rejected`) accept the expected run/incarnation and reject mismatches with `STALE_RUN` or `STALE_INCAR` before examining step/version or mutating persistent state. Two real-guest *targets* send wrong-run/wrong-incarnation forward messages; the updated **26-line** golden asserts two new `REJECTED` outputs. The deterministic negative `-10` success remains unchanged.

A directly addressable string message `cancel_failed` has been removed. The demonstration no longer accepts a `high_commit` command either: successful receipt of the second high-gradient contribution initiates the actor's own optimizer decision, and only an unsuccessful commit invokes the private compensator. Compensation independently requires complete gradient fan-in, an out-of-range candidate, and available step/version counters before retiring a rejected step. This removes two externally addressed optimizer-control commands but **does not authenticate ordinary string messages or gradient origin**. In particular, `run_id=101` and `incarnation=1` are fixed test values, not secure nonces, unforgeable identities, or restart-persistent runtime IDs.

**Required before production:** use runtime/supervisor-issued unique run/incarnation stamps; authenticate the authoritative sender (or transfer a nonforgeable CONTROL capability) before optimizer/compensation/termination admission; use a typed sealed command payload rather than the demonstration's `ActorMail<String>`; add message acknowledgments/CONTROL lifecycle verification; and replace `Result<int,String>` with an exhaustive nominal `TrainingError` once both backends accept it. Tests must reject forged identities *even when a sender guesses the scalar IDs*, replay after restart, and unauthorized `high_commit`/`after` commands. Merely including a matching run ID in an untrusted envelope is never authorization.

Current source and expected guest output are Git blobs `4597a2860fb5aaf0be5cf1dda5c16eaa93c5d8b2` and `029f88f5a174c1269a784ab4cc44b67e348156d1`, synchronized byte-for-byte to Java/GraalVM PR #1347. The ML host-only transaction model adds 8,192 seeded stale-identity/replay probes; this is not a compiler run or guest scheduling proof. Java source-owned runner path requires reviewed exact SHA/tree approval, a protected ref, and a private ephemeral runner. The private-source incident freeze remains in effect: no cross-org source mirroring. **Both PRs remain drafts pending native compiled tests.**

## Follow-up lifecycle hardening and independent CONTROL oracle

The demo receiver no longer accepts `after` as a standalone, externally addressed termination command. `new_negative_forward` now completes only when the final successful forward was admitted, both final-step forward contexts exist, the counters match the expected demo run/incarnation/step/version, and no gradients are pending. This is a **state predicate, not sender authentication**: any holder of the current `ActorRef<String>` can still spoof the data messages. The final output remains the updated 27 lines; the guest source Git blob is `4597a2860fb5aaf0be5cf1dda5c16eaa93c5d8b2`.

An independently authored Node host-only CONTROL capability oracle was added to funded sagitta-stack PR #31. The model uses unforgeable in-process object identity for a mock supervisor authority, distinct from public data messages, and tests forged/copy-prop capability requests, invalid optimization/discard, actor recreation with a new incarnation, and early termination rejection. It does **not** implement or certify these authority semantics in Oreslang. Production MUST move CONTROL onto the trusted runtime-delivered reserved lane, not a public sendable object or guessed run value. See P0 ML issue #11 and the open #22/#77 private-source export holds.

The Java source PR #1347 tracks the same guest blob and golden; no compiled Java/Graal/LLVM proof exists while source-org Actions jobs have zero steps. Do not represent a passing carrier model as native execution.

## Concurrent graph-fence audit reconciliation

A separate concurrent ML branch update added `graph_id = 202` and `STALE_GRAPH` to **all four** run/incarnation/version-fenced scalar operations, including a new deliberately wrong-graph forward. It inserted one additional rejection into the expected stdout (now **27 lines**), and the host-side scalar transaction test added graph-aware behavior. The previous lifecycle hardening was preserved. The latest ML guest fixture Git blob is `4597a2860fb5aaf0be5cf1dda5c16eaa93c5d8b2`; golden blob is `b602f9821817696f16499af1a74cb22d49a0ecab`. Java PR #1347's fixture, golden, JUnit assertions and both protected runner workflow hashes were reconciled to these exact blobs.

Graph/run/incarnation values remain *guessable demo IDs*, never authorization. The native parser/JUnit/guest gates have not executed, and the separate source-free funded carrier's 87/87 passing tests do not establish graph-fenced native runtime correctness.
