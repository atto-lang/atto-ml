# Bounded packed CPU KANLayer forward v1

Status: DRAFT; real Oreslang compiler/guest execution NOT_EXECUTED.

Stacked on multi-span cubic PR #15, itself stacked on KAN/MFKAN PR #14.
Native Oreslang source is distinct from host-only JavaScript test oracles.

## Source contract

Function: kan_layer_forward_f64_into(inputs, knots, coefficients, biases,
input_count, output_count, controls_per_edge, outputs, scratch) -> Result<bool,String>

- One shared clamped cubic knot grid; each input coordinate must be in its
  closed domain. Interior multiplicity <=3; no clipping/extrapolation.
- Input dimensions I=1..32, output dimensions O=1..32, controls C=4..256.
- Flat coefficient layout [O,I,C] with index (o*I+i)*C+k, no more than
  65536 total coefficients. This is a logical representation, NOT a second
  native Tensor/NdArray storage ABI.
- Outputs equal bias[o]+sum over input edges of spline(x[i],c[o,i,:]).
- All input/bias/knot/coeff values and computed edge/accumulation values
  must be finite and bounded in [-1e6,1e6].
- Caller supplies exact-size mutable outputs and scratch buffers. Borrowed
  inputs/knots/parameters must not alias either mutable buffer. Scratch and
  outputs may not alias.
- Inputs/descriptors/shape/domain are checked first. Scratch receives the
  complete result before any output writes; if any later edge fails, the
  destination is unchanged but the explicitly expendable scratch may have
  changed. Stronger concurrency/borrow guarantees require real compiler tests.
- This version implements forward only, NOT a full trainable KANLayer.

Native code: src/kan/clamped_cubic_bspline_f64.ores

## Evidence

tests/kan_layer_packed_cpu.test.mjs: nine HOST-ONLY tests, including 2,000
deterministic randomized multi-span layer comparisons, 256 controls, quotas,
malformed descriptors and values, alias rejection, clamped endpoints and
failure-atomic outputs when a later output overflows. The exact oracle code
passed 9/9 in isolated V8. This does not constitute an Oreslang compiler pass.

examples/kan_layer_f64_smoke.ores: proposed native guest test checking a 2x2
layer, bad shape, overflow after earlier successful accumulation, and alias
rejection. Expected exact stdout: ORES_KAN_LAYER_PASS.

scripts/verify-kan-layer-guest.sh requires a clean pinned git head, expected
kernel/fixture Git blobs, separately verified compiler SHA256, bounded GNU
timeout, empty stderr and exact stdout. There is no compiler available in
the current workspace. A caller-selected digest alone cannot authenticate a
compiler; use a trusted independently reviewed toolchain in actual CI.

The inherited scripts/verify-kan-multispan-guest.sh is repinned to the
extended kernel blob on this stacked branch. Any subsequent kernel changes
require repinning both. Zero-step Actions failures are NOT_EXECUTED, not
numerical failures or proof that all Actions minutes are exhausted.

## Promotion gates

1. Restore an authorized same-org Actions runner or independently approved
   dedicated private carrier, following Java security issue #1501:
   https://github.com/ores-truffle-oreslang/oreslang-source.java/issues/1501
   Private Git tree mirroring without this approval is not allowed.
2. Run exact Java/GraalVM guest compiler and LLVM compiler against the pinned
   fixtures, including static List mut borrow and Result checks.
3. Reconcile this logical flat packing with canonical NdArray ownership and
   shape/stride ABI; do not fork a new tensor store.
4. Add analytic multi-edge input/parameter VJPs and fan-in accumulation, then
   version-fenced optimizer state, SGD and held-out nonlinear regression.
5. Register operations in the generic model DAG without KAN grammar. There
   is not yet trainable KAN/MFKAN graph parity, GPU, AOT or hard RT evidence.
