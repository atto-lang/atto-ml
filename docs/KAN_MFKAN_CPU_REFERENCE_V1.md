# KAN / multi-fidelity KAN CPU reference — bounded scalar v1

**Status: source-authored; guest execution NOT_EXECUTED.** The standalone
Oreslang scalar kernels are not yet verified with the Java/Graal or LLVM
compiler. The JavaScript reference is a **host-only test oracle** and is
never part of the Oreslang runtime or production numerical implementation.

## Mathematics

One local **uniform cubic B-spline segment** for t in [0,1] has four basis
functions:

    B0 = (1-t)^3 / 6
    B1 = (3t^3 - 6t^2 + 4) / 6
    B2 = (-3t^3 + 3t^2 + 3t + 1) / 6
    B3 = t^3 / 6
    edge(t) = sum_k coeff[k] * Bk(t)

**This is only one local spline segment**, not full multi-interval Cox-de Boor
evaluation, trainable knot grids, or a complete KAN training implementation.
No implicit clipping/normalization, type conversion, broadcast, allocation,
device transfer or out-of-domain extrapolation is permitted.

For input dimension I and output dimension O, a KAN layer computes

    y[o] = bias[o] + sum_i edge(x[i]; coeff[o,i,0..3])

The host oracle implements exact shape and resource bounds, per-edge
coefficient gradients, and input gradients accumulated over every fan-in
edge. Tensor storage in real Oreslang must reuse the canonical
oreslang-data-science NdArray ABI; it must not fork another storage system.

## MFKAN: one explicit residual formulation

The low-fidelity KAN produces yL = lowKAN(x). The high-fidelity nonlinear
KAN and affine linear branch each receive concat(x,yL). Then:

    yH = yL + alpha * Linear([x,yL]) + (1-alpha) * KAN([x,yL])
    0 <= alpha <= 1

The residual and alpha placement are explicit design choices, not a
universally specified definition of MFKAN. Encode their exact semantics
in the operation ABI, implementation hash and model checkpoint.

The host reference propagates derivatives through both correction
branches, their dependency on yL, and the direct yL skip. Freezing the low
stage prevents updates to low parameters without erasing input gradients.
High-fidelity training labels must be paired to the exact low prediction,
graph/parameter version and normalization contract.

Because this profile supports only a local spline segment, **every x and
low prediction consumed as a spline coordinate must be in [0,1]**.
Reject any violation instead of clipping. Nonfinite values, invalid shape,
invalid alpha and out-of-quota dimensions also reject before computation.

## What this PR provides

- `src/kan/cubic_bspline_f64.ores`: pure scalar cubic forward, input derivative,
  coefficient derivative and residual MFKAN blend, returning Result errors.
- `examples/kan_mfkan_scalar_smoke.ores`: unexecuted guest-acceptance
  fixture, expected output `ORES_KAN_MFKAN_SCALAR_PASS`.
- `reference/kan_mfkan_reference.mjs`: independent Node 22 reference for
  a multi-edge KAN layer, MFKAN high/low composition and analytic VJPs.
- `tests/kan_mfkan_reference.test.mjs`: 11 host tests covering basis
  identities, finite differences, fan-in, freezing, toy loss reduction
  and fail-closed invalid numerical inputs.
- A source-local CI workflow limited to **HOST_REFERENCE_ONLY** checks.

Run the host suite:

    node --test tests/kan_mfkan_reference.test.mjs

## Forward migration to general-purpose define model

Do **not** add KAN/MFKAN keywords to the parser. Instead, admit vetted
KANLayer/ResidualBlend operations into the compiler-owned, general-purpose
typed ModelGraph with explicit port ABIs, resource bounds, implementation
fingerprints, optional VJPs and per-instance trainable state. The current
Java executable model draft is linear/pure/scalar-only; named-port DAG joins,
stateful parameters and autograd integration must be implemented first.

Next gates: actual pinned Java/Graal guest scalar execution; negative guest
validation; multi-knot spline/Cox-de Boor kernels and robust knot boundaries;
NdArray-backed multi-feature layers and VJP; compiler-lowered named-port DAG
composition on Java and LLVM; bounded actor-owned optimizer state and training;
AOT/GPU execution only after separately evidenced target support.
Zero-step CI jobs do not count as test passes; no private source may be
mirrored to another organization without authorization.

Trackers:
- https://github.com/ores-truffle-oreslang/oreslang-source.java/issues/1332
- https://github.com/ores-truffle-oreslang/oreslang-source.llvm/issues/621
- https://github.com/ores-truffle-oreslang/oreslang-common-source/pull/93
- https://github.com/ores-truffle-oreslang/oreslang-ml/pull/5
