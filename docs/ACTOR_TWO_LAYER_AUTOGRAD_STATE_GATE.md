# Actor-owned two-layer reverse-mode SGD: CPU milestone

**Status:** Proposed executable Oreslang guest fixture, NOT verified by any pinned Java/Graal or LLVM compiler. Host-only independent numerical/CAS oracle: 9 cases validated in isolated V8 on the exact authored test blob before review. No GPU, tensor allocation ownership, model parser, multi-actor pipeline, distributed synchronization or runtime RW-lock integration evidence.

## What was implemented in source

`examples/train_actor_two_layer_autograd.ores` declares one `TwoLayerTrainingActor` with exactly one actor-owned `state { ... }` holding **w1, b1, w2, b2, version and committed_epoch**. There are no ordinary actor fields and no `self.state` access.

An incoming `"fit"` mailbox message performs 1,000 iterations of the 2-layer ReLU function `y_hat=w2*ReLU(w1*x+b1)+b2`. Four CPU examples train toward `y=2*x+1`. On each iteration:

1. A **single read loan** copies all four f64 scalar weights and the epoch/version fences into an immutable primitive tuple, then releases the loan. No state loan or mutable backing storage escapes.
2. A separate `TapeF64` of 13 nodes runs `forward_checked_into` and `backward_checked_into` for all four examples **entirely outside the loan**, obtaining true tape VJPs. This is not handwritten differentiation in the Oreslang fixture.
3. Check finite gradient sums, loss, and *every* candidate weight before touching actor state.
4. A sealed `self.handle_state(critical write |s| -> ...)` requires exact pinned version and prior epoch, then publishes all four new parameters, next version and committed epoch in the **same short critical write transaction**. A competing calculation with the same pins is rejected as `STALE_SNAPSHOT`, not silently rebased.
5. After epoch 1000 the actor requires loss `<1e-6` and version `1001` before emitting exactly `ORES_ACTOR_TWO_LAYER_AUTOGRAD_PASS`. The caller uses `spawn`, `get_ready()`, a message, then `get_done()`, not direct actor method calls.

There is no optimizer momentum state in this SGD example. An AdamW variant must atomically publish moment slots and weights in the same state transaction; it is invalid to commit one without the other.

## Independent checks and evidence boundaries

`tests/actor_two_layer_autograd_state_contract.test.mjs` contains:

- a source-contract check for the sole state, read-copy and write CAS, and absence of compute in the critical write transaction;
- source checks for reverse-mode tape gradients and checked numerical values;
- central finite-difference checks for all four derivatives and a ReLU inactive branch;
- full 1,000-epoch host-reference convergence and exactly 1,000 publications;
- duplicate/stale step, mismatched epoch and old-weight snapshot cases;
- NaN/+Infinity/-Infinity rejection without partial publication;
- 128 deterministic competing candidate interleavings.

The source and JS oracle are **separate**. The JS oracle cannot prove the `.ores` program parses, that Graal or LLVM execute any tape node, that a real actor scheduler serializes writes, or that the returned tuple is accepted by the current ownership checker.

`scripts/verify-actor-two-layer-guest.sh` is a fail-closed **future executable-guest gate**. It requires exact, clean Git heads for the sibling `oreslang-ml` and `oreslang-autograd` worktrees and a SHA-256-pinned absolute runner, checks actual imported `.ores` files, rejects any guest failure, and diffs stdout against the single PASS line. Invocation (only after a compatible compiler build exists):

```bash
export ORESLANG_ML_EXPECTED_SHA=<exact-reviewed-ml-commit>
export ORESLANG_AUTOGRAD_EXPECTED_SHA=<exact-reviewed-autograd-commit>
export ORESLANG_RUNNER_SHA256=<sha256-of-absolute-runner-executable>
bash scripts/verify-actor-two-layer-guest.sh /absolute/path/to/oreslang
```

A runner hash alone does not pin dynamically loaded JARs, native libraries or GraalVM/toolchain artifacts. Certifying evidence must bind the compiler's full dependency provenance, source commit/tree, `.ores` file blob SHA, executed test logs and negative admission tests. **No private source export to another organization** while conformance security incidents #22/#77 and authorizations remain open.

## Remaining gates

**G1 — Java/LLVM compilation and admission.** Verify tuple destructuring, actor state loan/effect rules, `Result<float,String>`, TapeF64 imports, loop semantics and mailbox completion from actual `.ores` guest. Compiler failures must be fixed, not bypassed. The existing Java scalar guest PR #1347 is still blocked from demonstrated execution; its nominal enum parser prerequisite comes from the independent ADT PR #623.

**G2 — runtime and numerical correctness.** Run the exact fixture on real Graal and LLVM, compare stdout, test stale/duplicate concurrent admissions (up to three execution threads), cancellation after read but before commit, solver convergence and memory bounds. Record named production state gate implementation and tests, not only abstract JS models.

**G3 — tensor and pipeline promotion.** Replace primitive scalar snapshot with immutable owned parameter `TensorF64` leases; perform batched matmul and tensor VJPs outside loans. Partition into two or more actor stages; route checked forward/backward envelopes and bounded activations. Use coordinator-bound weight publication for sharded actors.

Do not upgrade this draft into an executed end-to-end Oreslang training claim until G1/G2 actually pass.
