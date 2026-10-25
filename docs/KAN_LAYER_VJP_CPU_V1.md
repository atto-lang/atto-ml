# Bounded packed KANLayer reverse-mode VJP (CPU reference v1)

**Status: DRAFT, compiled Oreslang guest NOT_EXECUTED.** This change is stacked on draft PR #17, then #15, then #14. The host Node/V8 test is a separate mathematical oracle, not proof of Java/LLVM compilation.

## Native operation

`kan_layer_vjp_f64_into(inputs, knots, coefficients, biases, upstream,
input_count, output_count, controls_per_edge, grad_inputs,
grad_coefficients, grad_biases, work_inputs, work_coefficients,
work_biases, probe_outputs, probe_scratch) -> Result<bool,String>`.

For I inputs, O outputs, C cubic controls per edge, and flat row-major parameter layout `[O,I,C]`:

- `grad_inputs[i] = sum_o upstream[o] * d_spline(x[i], coefficient[o,i,:])/dx`.
- `grad_coefficients[(o*I+i)*C+j] = upstream[o] * basis[j,3](x[i])`.
- `grad_biases[o] = upstream[o]`.
- All inactive coefficient gradients are zeroed. Each independent edge has its own control lane, while input gradients **accumulate fan-in across outputs**.
- Limits remain I/O 1..32, C 4..256, total controls at most 65,536, all admitted numerics and every intermediate grad magnitude <=1e6.
- At triple interior knots the right-hand derivative is selected, at the right endpoint the left-hand derivative. Both inherit the spline ABI from #15.
- Caller owns **six unique** gradient destinations/scratch arrays and **two unique** forward-validation probe arrays. Every output/work buffer must be distinct from every read source and from every other writable buffer. The unique borrow and parameter snapshot properties still require actual Java/LLVM type-check and runtime evidence.
- An invocation starts with bounds/dimension/alias/gradient-output-shape checks, then calls the same native forward function for descriptor, knot, input-domain, and forward overflow validation. Only caller-owned *expendable* probes are overwritten by this preflight.
- After computing all checked input, coefficient and bias gradients into work buffers, all three published gradient destinations are copied. No destination is written on an ordinary Result error; work buffers and probes may be modified on error.
- **One synchronous sample only.** There is no cross-microbatch accumulation, parameter-version enforcement, Tensor/NdArray ownership, GPU, actor suspension, hot reload or native optimizer update in this contract.

## Source and verification

- Native source: `src/kan/clamped_cubic_bspline_f64.ores`, Git blob `f49cc2ef36f808a18c5304c2a536497bb8d4759b`.
- Guest acceptance fixture: `examples/kan_layer_vjp_f64_smoke.ores`, Git blob `fd1a9f1d2a535d632e4b76da21850683418a5177`; exact success stdout `ORES_KAN_LAYER_VJP_PASS`. Tests 2x2 coefficients/bias/input, input overflow preserving *all three* destinations, and an alias rejection.
- Runner gate: `scripts/verify-kan-layer-vjp-guest.sh`, demands clean exact source HEAD, reviewed source and fixture Git hashes, independently pinned SHA256 of a real Oreslang executable, bounded timeout, empty stderr, and exact stdout.
- **8/8** functions of the independent Cox-de Boor host oracle passed in the V8 test/assert harness before commit; suite `tests/kan_layer_vjp_cpu.test.mjs`, blob `6c392f431af932343892e458cbd04e3b2952699d`. Includes 1,200 seeded finite-difference comparisons, single-lane and fan-in overflow rejection, alias/shape/NaN checking, right-hand triple-knot derivative, zero upstream, and a **host-only** 20-step SGD loss-reduction illustration.
- The host SGD illustration deliberately updates a local JS array to test *gradient usefulness*. It is **NOT** a compiled Oreslang SGD/optimizer or an end-to-end native training claim.

## Required next gates

1. Obtain an authorized source-local runner; if considering other GitHub organizations, no private tree transfer without a reviewed private carrier ID, snapshot contract, and independent security approval as documented in [source Java issue #1501](https://github.com/ores-truffle-oreslang/oreslang-source.java/issues/1501). Zero-step workflow failures are NOT_EXECUTED, not compiler or numerical failures.
2. Run actual exact-head Java/Graal and LLVM guests under pinned compiler executable identity; check all long parameter headers, List mut borrows/alias handling, failure paths, and no implicit heap on critical paths.
3. Reconcile logical flat coefficient packing with one canonical owned NdArray/TensorF64 storage ABI and the actor's immutable parameter-snapshot lease.
4. Implement version-fenced gradient accumulation and an optimizer-owner SGD update with candidate preflight/commit atomicity. Train a full nonlinear data set, not only one host regression example.
5. Integrate these operations into generic typed `define model` DAG lowering for both compiler backends; keep model DSL independent of KAN-specific syntax.

**Remain draft and do not merge until execution produces real nonempty CI steps.**
