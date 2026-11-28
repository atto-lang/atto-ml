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
- `scripts/verify-kan-multispan-guest.sh`: fail-closed actual guest verifier requiring clean exact source commit, fixed checked Oreslang source blobs, absolute compiler path + SHA256, GNU timeout, stderr-free exact stdout. It does not run without a compiler.
- `tests/kan_multispan_reference.test.mjs`: independent Node source-free numerical tests, 10,000 seeded de Boor versus Cox-de Boor comparisons, 2,000 finite-difference derivative trials, 800 complete coefficient-gradient fixture sweeps, knot endpoint/multiplicity tests, right-hand derivative at a triple interior knot, bounds and malformed-input rejection.

```sh
node --test tests/kan_multispan_reference.test.mjs
```

A local Node 22 test development copy passed **10/10** cases. The exact committed test file must still be independently verified against the active PR head. That alone is not proof the committed Git blob executed under GitHub Actions; record exact-head checks independently.

## October 25, 2026: input-VJP and exact-blob audit

- Public `kan_clamped_cubic_vjp_input_f64` now consumes a separately
  range-checked upstream gradient, evaluates a checked local input derivative,
  and rejects the **product** when it exceeds the bounded f64 profile.
  An accepted 3.0 derivative times an accepted 1,000,000 upstream produces
  3,000,000 and must fail closed. This does not publish an optimizer update.
- `examples/kan_multispan_cubic_smoke.ores` now checks successful positive
  input VJP, overflow rejection, upstream rejection, and a triple-interior-knot
  right-hand derivative distinct from the left derivative.
- Actual guest verifier pins exactly kernel blob
  `9413bab298e04a6d0cf006711c59d71ebee91585` and fixture blob
  `93732a26c708c1e06604c6b5d1127ed87fe3ba28`.
- A newly committed source-free test
  `tests/kan_multispan_input_vjp.test.mjs` (blob
  `d2c9fb7c6cfe51d57f4b6c1862dd02699a92787a`) passed **5/5**
  tests on local Node 22.16.0, including 2,000 finite differences.
- Both exact committed host suites, totaling **18/18** test functions,
  passed inside an isolated V8 harness. That is distinct from native
  Node 22 execution of both suites together and is **not** Oreslang guest
  compilation.
- Source-org GitHub Actions was unable to start runner steps at the last
  observed head; such runs are **NOT_EXECUTED**, not numerical failures.
- `oreslang-source.java` protected-governance private-CI destination
  allowlist contains no approved numeric-ID destination; a comment-only
  candidate is **not** an authorization to mirror private source. Do not
  use direct connector tree writes or an unrelated private test repository
  as a workaround.

## Compilation and promotion gates

1. Run the unchanged .ores smoke fixture with an exact pinned Java/GraalVM Oreslang compiler, compare stdout and stderr, and test negative rejection **inside** the compiled guest.
2. Run full `mvn verify` and real guest compiler tests on an authorized runner with nonzero job steps; confirm all imports and result/loop syntax rather than inferring validity from JS.
3. Validate the identical fixture on LLVM and compare numeric outputs, diagnostics and ABI/provenance. Do not claim parity from host oracle tests.
4. Add canonical owned `NdArray` tensor storage and gradient accumulation with mutation/borrow/version fences, then wire trusted KAN operations into the general-purpose `define model` DAG lowering.
5. Extend testing to adversarial conditioning near extremely close knots, continuity classes at repeated knots, deterministic CPU f32/f64 behavior and target-specific resource measurement. This is not a proven hard-real-time path.

**CI security:** Neither GitHub Actions exhaustion nor spare minutes are inferred from zero-step failures. Cross-org mirroring of private source requires an approved *dedicated private destination* reviewed under [Java #1501](https://github.com/ores-truffle-oreslang/oreslang-source.java/issues/1501), not just a successful source-free canary. No source transfer is authorized by this PR.

Parent implementation: [oreslang-ml PR #14](https://github.com/ores-truffle-oreslang/oreslang-ml/pull/14).
