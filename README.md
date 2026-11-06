# Oreslang ML

Oreslang-native neural network layers, losses, optimizers and training loops on top of `oreslang-tensor` and `oreslang-autograd`.

**Current status:** `main` is contract/metadata-only; native ML examples and
kernels are under review on stacked draft PRs. There is no compiled-guest
Java/LLVM release proof. See [security and release gates](docs/ML_SECURITY_RELEASE_GATES.md)
before using any prototype for production or claiming backend parity.

## P0 acceptance
- CPU-only reference model, linear forward/loss/SGD and a loss-decreasing training example.
- No hidden Python, Java or vendor ML framework implementation.
- Data and optimizer state have explicit ownership and lifecycle.

See `docs/CONTRACT.md` after bootstrap PR.

## KAN / multi-fidelity KAN v1 (proposal)

`src/kan/cubic_bspline_f64.ores` introduces pure Oreslang **single-span cubic**
KAN scalar operations and analytic derivative helpers. The candidate guest
example is `examples/kan_mfkan_scalar_smoke.ores`.

`reference/kan_mfkan_reference.mjs` is an independent **host-only numerical
oracle**, not an ML runtime, with shape-checked multi-edge layers, explicit
multi-fidelity correction and gradients. Run:

```sh
node --test tests/kan_mfkan_reference.test.mjs
```

See [`docs/KAN_MFKAN_CPU_REFERENCE_V1.md`](docs/KAN_MFKAN_CPU_REFERENCE_V1.md)
for the mathematical contract, out-of-domain policy, and implementation gates.
Java/LLVM compiled KAN training and `define model` DAG composition are **not
claimed** by the host-only tests.
