# Multi-span clamped cubic B-spline (source-owned CPU candidate)

**Implementation status:** Oreslang source authored on a stacked draft branch; **compiled guest execution NOT_EXECUTED**. The independent Node 22 numerical suite is **host-only** and is not a compiler, runtime, or backend-parity receipt.

## Scope and semantics

- The degree is **exactly 3**. A coefficient vector of length `n` (4 to 256 inclusive) requires exactly `n+4` nondecreasing finite knots.
- The first four knots are equal to `a=knots[3]`; the last four equal to `b=knots[n]`, with `a < b`. Interior knots must lie strictly inside `(a,b)`. Interior multiplicity may be 1, 2 or 3; 4 is rejected.
- The domain includes both endpoints, `[a,b]`. Evaluation at `b` selects the last span and yields the last coefficient. At a repeated interior knot, the **right-hand span** is selected. Thus the input derivative at a nonsmooth interior knot is the right-hand derivative; at `b` it is the left-hand derivative. This rule is part of the operation ABI.
- All knots, coefficients, inputs, outputs, and derivatives are finite bounded values with magnitude at most `1e6`. Out-of-domain evaluation fails. No hidden clipping, extrapolation or normalization.
- Fixed 4-scalar iterative de Boor workspace for forward evaluation; fixed 3-scalar quadratic de Boor workspace for the derivative. All loops and knot access are bounded by validation before indexing. Spline length and coefficient count are static *runtime-bounded*, not verified freestanding or allocation-free at every compiler/target.
- Invalid knot descriptors, unsupported multiplicities, nonfinite values, size mismatch, coefficient index errors, degenerate active denominators and oversized derivative intermediates all fail closed with typed `Result<float,String>` errors.

For cubic basis `N[i,3](x)` and coefficients `c[i]`, evaluation is
`f(x) = sum_i c[i] * N[i,3](x)`. The analytic input derivative uses reduced degree-2 coefficients:

```
q[i] = 3*(c[i+1]-c[i])/(k[i+4]-k[i+1])
f'(x) = degree2_deboor(x, trimmed_knots, q)
```

The coefficient VJP for coefficient `i` is `upstream * N[i,3](x)`, zero outside the four active controls. The library returns one lane at a time; accumulation and parameter publication remain responsibilities of a caller with correct ownership, forward-activation identity and parameter-version fencing.

## Source files / acceptance

- `src/kan/clamped_cubic_bspline_f64.ores`: candidate native Oreslang forward/input-derivative/coefficient-VJP implementation, requiring no native numerical framework.
- `examples/kan_multispan_cubic_smoke.ores`: compiled guest acceptance fixture. Exact stdout marker: `ORES_KAN_MULTISPAN_PASS`. Includes both boundaries, midpoint, derivative, an inactive coefficient and negative descriptors.
- `tests/kan_multispan_reference.test.mjs`: independent Node source-free numerical tests, 10,000 seeded de Boor versus Cox-de Boor comparisons, 2,000 finite-difference derivative trials, 800 complete coefficient-gradient fixture sweeps, knot endpoint/multiplicity tests, bounds and malformed-input rejection.

```sh
node --test tests/kan_multispan_reference.test.mjs
```

A local Node 22 test development copy passed 9/9 cases. That alone is not proof the committed Git blob executed under GitHub Actions; record exact-head checks independently.

## Compilation and promotion gates

1. Run the unchanged .ores smoke fixture with an exact pinned Java/GraalVM Oreslang compiler, compare stdout and stderr, and test negative rejection **inside** the compiled guest.
2. Run full `mvn verify` and real guest compiler tests on an authorized runner with nonzero job steps; confirm all imports and result/loop syntax rather than inferring validity from JS.
3. Validate the identical fixture on LLVM and compare numeric outputs, diagnostics and ABI/provenance. Do not claim parity from host oracle tests.
4. Add canonical owned `NdArray` tensor storage and gradient accumulation with mutation/borrow/version fences, then wire trusted KAN operations into the general-purpose `define model` DAG lowering.
5. Extend testing to adversarial conditioning near extremely close knots, continuity classes at repeated knots, deterministic CPU f32/f64 behavior and target-specific resource measurement. This is not a proven hard-real-time path.

**CI security:** Neither GitHub Actions exhaustion nor spare minutes are inferred from zero-step failures. Cross-org mirroring of private source requires an approved *dedicated private destination* reviewed under [Java #1501](https://github.com/ores-truffle-oreslang/oreslang-source.java/issues/1501), not just a successful source-free canary. No source transfer is authorized by this PR.

Parent implementation: [oreslang-ml PR #14](https://github.com/ores-truffle-oreslang/oreslang-ml/pull/14).
