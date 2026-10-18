# Supervised CPU data → tensor → autograd integration (P1 gate)

**Status: proposed real-guest source fixture and independently tested HOST-ONLY math oracle. Not yet compiled or run by Oreslang.** No private-source cross-organization copy is authorized by this work.

## Scope and real dependencies

`examples/train_supervised_tensor_autograd.ores` uses four existing Oreslang sources through their public library functions, not a surrogate host implementation:

- `oreslang-data-science`: `FloatColumn`, `compact_supervised_f64_into` (nullable feature/label mask intersection, original row IDs), and `scatter_add_f64_into` (reverse row provenance)
- `oreslang-tensor`: `TensorF64`, `matmul_2d_f64_into` ([N,1] @ [1,1])
- `oreslang-autograd`: `TapeF64`, `forward_checked_into`, `backward_checked_into` (genuine bounded reverse-mode scalar arithmetic graph)
- `oreslang-ml`: bounded 500-step CPU SGD demonstration, weight and bias, training convergence and tensor/tape forward equivalence.

A fifth invalid source row deliberately contains extreme values and must not be admitted as a training example. Initial gradient provenance is scattered to the five original rows, ensuring the filtered row remains zero. Four accepted source rows form exactly [4,1] tensor input storage: the **current List-backed TensorF64 is not a dynamic zero-copy view of a partially used buffer**. P0 explicitly rejects pretending otherwise.

The tape has 8 nodes per example: `w`, `x`, `b`, `-target`, `w*x`, `+b`, `+(-target)`, `square`. Gradient contributions to `w` and `b` come from `backward_checked_into`, never handwritten guest gradients. Source code cross-checks the forward activation against the tensor matrix multiplication plus bias. The SGD update computes both candidates before mutating the ordinary synchronous routine's weight/bias lists.

Expected stdout is exactly one line: `ORES_ML_SUPERVISED_TENSOR_AUTOGRAD_PASS`.

## Reproducible real-guest gate

Place checked-out sibling repositories under one root:

```text
workspace/
  oreslang-ml/
  oreslang-data-science/
  oreslang-tensor/
  oreslang-autograd/
```

With an **already available audited Oreslang compiler**:

```sh
ORESLANG_BIN=/absolute/path/to/oreslang ORESLANG_BIN_SHA256=<64-hex-digest> bash scripts/verify-supervised-cpu-guest.sh
```

The script refuses an absent runtime or sibling source and requires the **exact Oreslang executable SHA-256** and pins **six actual Git blob identities**, validated with `git hash-object` for the current working files. It does not fetch/checkout/copy source, create a runner, invent a pass, or use GitHub credentials. It calls the real guest program, compares exact stdout and returns nonzero on compile/runtime/output failure. The source-org's current Actions inability to allocate runners is neither a pass nor proof of compiler failure.

The host test `tests/supervised_tensor_autograd_contract.test.mjs` is independent JavaScript numerical validation and syntax/composition inspection only, **not an implementation of Oreslang library APIs**. It should run under Node 22, and its results are labeled HOST_ONLY. An isolated V8 check covered 11 cases at the first reviewed source/test Git blobs; no native guest result is available yet.

## Actor boundary and required next gate

This P1 fixture is intentionally synchronous. It is **not** a disguised multi-actor trainer. The existing scalar actor example on PR #7 separately exercises mailbox-only interaction and short `state` loans. Merging the two into true actor-native training requires:

1. Each neural-network actor has one `state { ... }` and **no ordinary persistent instance fields**. Own tensor/version handles, optimizer slots, saved-forward contexts and edge receipts underneath that sole schema.
2. `with state.read, state.write` declares effect bounds; actual state admission must go through lexical `using state.read` and `using state.write(critical ...)`, or the sealed `self.handle_state(critical read/write ...)` intrinsic.
3. Short read/write loans pin independently immutable tensors; tensor matmul and VJP execute **outside** loans. No `await`, channel wait, I/O, GPU submit, or actor send with an active state loan.
4. Versioned envelopes bind **run, graph generation, step, actor incarnation, microbatch, edge identity and parameter version**. Forward context pins the same weight version used in backward. Duplicate/late gradients and partial fan-in cannot trigger an optimizer commit.
5. Prepared candidate weights/optimizer state publish in one local exclusive transition, with a replay-safe receipt. Cross-shard/distributed publication requires additional protocol and consistent recovery; a single actor's RW gate cannot make a global transaction.
6. Train two stages using actual typed actor messages; run 1 and 3 execution threads with bounded memory, cancellation, poison/stale messages, lost sends and deterministic replay. Only after this passes do GPU execution and pipeline scheduling qualify.

### Current evidence tiers

- E0: independent host numerical/source tests, no Oreslang guest execution (this PR).
- E1: real `.ores` execution with all six exact source blob IDs and source-tree/CLI provenance, no compiler bypass.
- E2: typed multi-actor forward/backward training, locked state-loan and version consistency under concurrency.
- E3: Java Maven/JDK25, JVM/native-image, LLVM native + sanitizers, bounded-memory/failure-injection and GPU if applicable.

Source-private cross-organization mirror work remains gated by the open [public carrier incident #22](https://github.com/ores-truffle-oreslang/oreslang-platform-conformance/issues/22) and [private carrier incident #77](https://github.com/ores-truffle-oreslang/oreslang-platform-conformance/issues/77); no new private-tree transfers are permitted merely because an alternative runner has paid GitHub Actions capacity.

## Experimental actor forward/backward control-path fixture

`examples/two_stage_actor_vjp_bootstrap.ores` is a separate experimental actor communication gate adapted from the already present LLVM transform-pipe syntax. It uses three actors: `InputWeight` → `HiddenWeight` → `SquaredLoss`, with a return gradient through `self.backward`. Each actor has exactly **one** guest-owned `state { ... }`; there are no actor-owned ordinary fields. Every state read/write is inside `using state.read` or `using state.write(critical ...)`, with `with state.read, state.write` effects declared. Forward and backward actor sends happen after state loans have closed. Local state records old weights, the saved forward activation, single-step completion receipts and version increments, rejecting local duplicate updates.

With the fixed int example `x=3`, `w0=2`, `w1=3`, target `16`, forward gives `y=18`; the loss upstream gradient is `4`; `dW1=24` and `dW0=36`. The source uses `ActorUtils.pipe` and `expect_pipeline(source, 36)` as an **unverified real-guest acceptance target**. The independent host contract runs 8 structure/calculus checks against this source. Integer gradients are deliberately hand-coded here: **this does NOT claim built-in tensor/autograd reverse graphs are connected to pipe actors**. That integration is still required before multi-stage NN training is achieved.

**Known correctness limitation:** the two trainable actors currently commit locally on gradient receipt, so a crash between downstream and upstream commits can leave inconsistent versions. This fixed, single-sample bootstrap is a control-flow demonstration, NOT a crash-atomic multi-actor optimizer. True training needs prepared candidate weights, a coordinated step manifest, durable recovery/fencing, graph/step/microbatch identifiers on messages, and clean cancellation of all staged candidate states. Do not use the bootstrap as a production optimizer.

Do not equate an isolated JavaScript host simulation with native LLVM/JVM actor or RW-gate execution. The underlying runtime, parser, actor mail delivery and dynamic graph topology need exact-source compiled tests before promotion.


## Hardening checkpoint: executor identity and atomic backward read (October 18)

- The compiler verifier now rejects a missing/nonabsolute/nonexecutable Oreslang binary, an absent or malformed SHA-256 pin, mismatched executable bytes, missing dependency worktrees, and any of six mismatched Git blobs **before** running the guest. The binary hash verifies the executable file itself; it is not a guarantee of signatures/trust, dynamically linked dependency identity, or the absence of in-place mutation after verification.
- The experimental actor pipe's hidden stage now copies `(saved_input, old_weight, version, issued, applied)` under **one logical state.read admission** rather than two independent read loans. This prevents mixed-generation input/weight observations if concurrent actor executions are enabled. Publication uses the observed version as its comparison value. The tuple-form source remains a proposed parser/compiler acceptance target, not proof of executed Oreslang support.
- Independent source and calculus host checks pass 12/12 for CPU integration and 10/10 for the actor pipe in an isolated JavaScript harness; those checks do not exercise the actual Oreslang interpreter or state lock.
- A separate **source-free** alternative-org workflow, `sagitta-stack/sagitta-infra` PR #31, [run #39155558378](https://github.com/sagitta-stack/sagitta-infra/actions/runs/39155558378), completed 39/39 Node host-only tests at pinned SHA `703f685b28ac1132af59e460a28fb1ad063b1f2f`. The tests include the independently authored interleaving model that demonstrates why split reads are unsafe. Log markers explicitly say Oreslang guest execution and private-tree mirroring did NOT occur.

**Still open:** source-org worker allocation, compiled Java and LLVM tests, numeric guest stdout, compiler-owned immutable tensor leases, actual multi-thread RW gate integration, per-message run/graph/step/actor epoch identity, and distributed commit/recovery.
