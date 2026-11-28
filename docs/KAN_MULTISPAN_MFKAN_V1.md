# Source-owned multi-span multi-fidelity KAN (CPU scalar candidate)

**State:** Draft proposal and Oreslang source; `GUEST=NOT_EXECUTED`. This document is not a Java/Graal, LLVM, AOT, training or ModelGraph-DAG parity claim.

## Source topology

Three independently checked degree-3 clamped cubic spline edges are evaluated at:

- `low(x)` — the low-fidelity predictor;
- `feature(x)` — the high-fidelity nonlinear branch on the original feature;
- `prediction(low(x))` — the other high-fidelity nonlinear branch, sharing the low output.

The high prediction is

```text
linear = wx*x + wl*low(x) + bias
nonlinear = feature(x) + prediction(low(x))
high = low(x) + mix*linear + (1-mix)*nonlinear, mix in [0,1]
```

This is explicit source composition; it does **not** mean `define model` can yet lower general named-port DAGs. Nor does it implement a tensor KAN layer or a multi-fidelity optimizer.

## Gradients and freeze contract

For `v=upstream` and `dP=prediction'(low)`, define:

```text
lowAdjoint = 1 + mix*wl + (1-mix)*dP
dHigh/dx * v = v*(lowAdjoint*low'(x) + mix*wx + (1-mix)*feature'(x))
dHigh/dLowCoeff[i] * v = lowAdjoint * v * basis_low[i](x)
```

`freeze_low=true` returns **zero parameter gradient** for the selected low coefficient after validating the forward operation, upstream and coefficient lane. It does **not** drop the low-stage contributions to the input gradient, or suppress other trainable branch gradients. The caller must freeze actual parameter publication separately; these functions do not implement optimizer ownership or updates.

All operation inputs/outputs are bounded finite f64 values (magnitude <= 1e6), no implicit domain extrapolation, normalization, tensor transfer or allocation-free claim. Each spline delegates knot order, multiplicity, shapes, domain and active derivative policy to `clamped_cubic_bspline_f64.ores`. At a nonsmooth repeated interior knot, the underlying ABI selects the right-hand derivative.

## Verification inventory

- `src/kan/mfkan_clamped_cubic_f64.ores`: proposed actual Oreslang scalar functions.
- `examples/kan_multispan_mfkan_smoke.ores`: proposed compiled guest acceptance target: exact stdout `ORES_MFKAN_MULTISPAN_PASS`, no stderr. Fixed golden: forward 0.825, input gradient 0.875, low coefficient #2 gradient 0.55, frozen coefficient 0; invalid mix, prediction domain and frozen invalid index fail.
- `tests/kan_multispan_mfkan_math.test.mjs`: independent **host-only** Cox-de Boor numerical oracle with a seeded forward corpus, finite-difference input and low coefficient VJP checks, frozen-low semantics and rejection cases. This does not execute the Oreslang functions.
- `scripts/verify-kan-multispan-mfkan-guest.sh`: real-guest verifier, pins exact clean source commit, reviewed Git blobs (base spline / MFKAN source / guest), absolute compiler executable SHA256 and exact stdout/stderr. Missing compiler reports `NOT_EXECUTED`, never PASS.

## Required gates before merge

1. On a permitted source environment, run exact-head Node 22 host numerical tests and `bash -n` on the verifier; record real nonempty CI steps. An isolated V8 host adapter is not Node CI.
2. Independently pin a working Oreslang Java/Graal compiler and its SHA256; execute `scripts/verify-kan-multispan-mfkan-guest.sh` on a clean exact source HEAD. Investigate any compile, import, float ABI, Result, list/ownership or runtime errors. Do not infer syntax support from source inspection.
3. Run the same compiled guest and independent numeric/gradient fixtures under LLVM once the native f64/List/Result ABI supports it, then run fuzz/ASan/UBSan where applicable. Native LLVM i64-only ModelGraph is insufficient evidence today.
4. Only after source execution, wire canonical `NdArray<T>` coefficient storage, caller-owned gradient accumulation, version-fenced training and general-purpose `define model` named-port DAG composition.

Neither zero-step GitHub Actions failures nor source-free alternate-org canaries authorize cross-organization private-source export or prove functional parity. Keep this stacked PR in DRAFT.
