# Four-repository nullable tensor + two-layer autograd CPU gate

**Status:** Oreslang guest source committed and source-only/numerical oracles checked. **Java/Graal/LLVM guest execution NOT VERIFIED.** No claim of actor-native tensor scheduling or GPU readiness.

## Source entrypoint

`examples/train_nullable_two_layer_autograd.ores` is a complete **source-level** reference, with imports into sibling directories named `oreslang-data-science`, `oreslang-tensor` and `oreslang-autograd`:

1. Select only jointly valid supervised rows and preserve source row positions `[0,2,3,4]`; original invalid row 1 is ignored.
2. Build a rank-2 `TensorF64` descriptor `[5,1]` over source input values. Gather the selected four rows into a separate `[4,1]` output with the tensor library's checked primitive; assert equality with columnar compaction.
3. Construct a topologically ordered 13-node reverse-mode `TapeF64` for `prediction = w2 * ReLU(w1*x + b1) + b2` and `loss = (prediction - y)^2`. VJP inputs and all four trainable parameters are read from the tape (not handwritten analytic gradients).
4. Train for 1000 full-batch SGD epochs, starting at `[w1,b1,w2,b2] = [1,1,1,0]`, with learning rate 0.02. Validate **each intermediate gradient/loss** and **all four** optimizer candidates before publishing any parameter.
5. Retain the final epoch's tape-computed input VJPs and use `row_gather_vjp_with_saved_f64_checked_into` to scatter back to the original five rows. Reject if invalid row 1 receives a gradient or the four expected row contributions do not match the saved input VJPs. Require final MSE below `1e-6`.
6. Emit **only** `ORES_TWO_LAYER_AUTOGRAD_PASS` on stdout on success. Every error emits a different marker.

The data and tensor buffers here are mutable caller-owned `List<float>`, **not** frozen backing storage, device leases, zero-copy Arrow buffers, or automatically sendable actor messages. Saved-forward equality is a value check, not an incarnation/generation identity proof. The single-process program is sequential; it is not evidence of production multi-actor pipeline execution.

## Stateful actor promotion — mandatory design boundary

The next actual actor-native stage stores all guest-owned persistent weights, optimizer slots, activation/gradient ledgers, fencing counters, and versioned parameter handles **inside its sole** `state { ... }` schema. No ordinary actor instance fields or `self.state` bypass.

- `with state.read` / `with state.write` declare maximum allowed effects; they DO NOT acquire locks.
- `using state.read`, `using state.write(critical ...)`, or sealed `self.handle_state(critical read/write ...)` acquire verified lexical access to that one actor's reader-writer gate.
- A short state loan may pin an independently immutable parameter storage version; that loan must end **before** tensor evaluation, actor sends, `await` or GPU work. Results acquire an exclusive critical loan for complete gradient fan-in and local version/epoch/step/graph-fenced publication.
- Multiple actors do not share an atomic local critical section. A coordinated, durable generation/commit protocol is required for distributed parameter-shard publication.
- Every backward envelope includes graph/run/step/edge/microbatch/incarnation and the **original forward weight version**, so stale, duplicate, orphan and partially accumulated contributions are rejected.

## Reproducible guest gate

`scripts/verify-two-layer-tensor-guest.sh` requires all four **independent local Git worktrees** at explicit immutable expected commit SHAs, no dirty/untracked files, all imported source paths, and an absolute runner executable with a verified SHA-256. The invocation is:

```bash
export ORESLANG_ML_EXPECTED_SHA=<exact-ml-head>
export ORESLANG_DATA_EXPECTED_SHA=<exact-data-head>
export ORESLANG_AUTOGRAD_EXPECTED_SHA=<exact-autograd-head>
export ORESLANG_TENSOR_EXPECTED_SHA=<exact-tensor-head>
export ORESLANG_RUNNER_SHA256=<sha256-of-runner-executable>
bash scripts/verify-two-layer-tensor-guest.sh /absolute/path/to/oreslang
```

No compiler is downloaded or built by the gate. The binary checksum does **not** cover dynamically loaded JARs, native shared libraries, JDK build, CLI flags or external environment state. A certifying CI receipt must also bind those dependencies and a frozen Git tree for every repository. Failed source syntax, unsupported imports, incorrect type/borrow/effect rules or output divergence must fail CI; do not weaken the checker to make the example pass.

## Verification and dependencies (2026-10-18)

| Source branch / PR | Head when fixture was authored | Responsibility |
| --- | --- | --- |
| `oreslang-data-science` PR #10 | `3d9d273a8fd705c321d60d281bd4ced88afb55b6` | Supervised nullable compaction |
| `oreslang-tensor` PR #5 | `89cb4ceea35397e970434ee4a1ba2b541f9bbd76` | Rank-2 gather + gradient scatter |
| `oreslang-autograd` PR #5 | `455fd9ffb403bb25614103ce6a101cba43e63c2c` | Scalar tape + saved-forward gather VJP |
| `oreslang-ml` PR #7 | See current branch head | Two-layer training integration |

`tests/nullable_two_layer_autograd_contract.test.mjs` exercises independent host-only 13-node tape semantics, all parameter/input finite differences, nullable row provenance, fixed SGD convergence, ReLU dead regions, duplicate-index gradient accumulation and atomic optimizer candidate validation. `tests/two_layer_runner_provenance_contract.test.mjs` tests the fail-closed runner source/snapshot checks. Passing these **does not** prove Oreslang guest compilation.

An independent funded source-free GitHub Actions workflow in `sagitta-stack/sagitta-infra` passed 31 synthetic mathematical/protocol cases on run [39154221794](https://github.com/sagitta-stack/sagitta-infra/actions/runs/39154221794), pinned at carrier commit `cca946b0402de65717db3f670b1bb4d4903c63d1`. It explicitly copied no Oreslang source and cannot certify the private Oreslang runtime. Source-org Actions and real guest compilation remain independently blocked/unverified. Open security incidents `oreslang-platform-conformance` #22 and #77 prohibit new cross-org private source mirrors until separately authorized.
