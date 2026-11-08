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

## Repository ownership and migration (2026-11-08)

**Current:** `ores-truffle-oreslang/oreslang-ml` in the old Oreslang org. **Public language:** AttoLang. Retain existing `.ores` files and historical repository/package/CLI aliases until compiler-backed `.atto` support and consumers have been validated.

- **This repo owns:** portable machine-learning algorithms and model-side APIs built on AttoLang tensor/autograd.
- **Not owned here:** commercial hosted model serving, registry or metered billing.
- **Organizations:** [atto-lang](https://github.com/atto-lang) is for language/compiler/tooling, portable ML/scientific libraries, conformance and demonstrations. [atto-co](https://github.com/atto-co) is for independently governed cloud hosting, deployment, tenant identity, billing and commercial services. Similar `atto-*` repo names do not imply transferred source ID or content parity.
- **Migration proposal:** identity-preserving GitHub transfer under the original repository name to `atto-lang/oreslang-ml` after reviews. No legacy-org direct move to `atto-co` is currently authorized. Later commercialization spinouts need their own signed off runbook.

**Repository transfer gates:** coordinate a single owner/rollback plan at [issue #26](https://github.com/ores-truffle-oreslang/.github/issues/26); consult private [runbook #25](https://github.com/ores-truffle-oreslang/.github/pull/25). Record exact source commit SHA/tree and numeric ID, target visibility and protection/rulesets, source/target admin and independent security approvals, PRs/tags/releases, packages/import consumers, webhooks, GitHub Apps, Actions/billing/runners, secrets and Pages. Never mirror private source without separate approval. After actual GitHub transfer verify identical ID/history/tree, restore protections, update references independently and run exact-head CI with nonzero executed steps. `steps=[]`/`null` is `NOT_EXECUTED`; host/static tests cannot establish compiled AttoLang numerical or guest execution parity.
