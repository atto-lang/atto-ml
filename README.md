# Oreslang ML

## AttoLang migration and current documentation (2026-11-18)

**AttoLang (formerly Oreslang)** is the current language brand. This `ores-truffle-oreslang/oreslang-ml` checkout is a historical organization location; the corresponding new-home project is **[atto-lang/atto-ml](https://github.com/atto-lang/atto-ml)**. Do not assume a same-named project has identical Git commits, issues, PRs, Actions or feature status without checking exact history. Keep the historical example commands below for this checkout until independently validated replacements exist.

**Source of truth:** [atto-lang/atto-spec](https://github.com/atto-lang/atto-spec) defines current and proposed language semantics; [atto-platform-conformance](https://github.com/atto-lang/atto-platform-conformance) provides portable positive/negative guest tests; [atto-common-source](https://github.com/atto-lang/atto-common-source) owns shared portable `.atto` code. New source aims for `snake_case` callables, `borrow!`/`take!`/`copy!`/`copy!!`/`share!`, [implicit borrowing](https://github.com/atto-lang/atto-spec/blob/main/language/implicit-argument-borrow-v1.md), [file/module constants with guarded one-time `os.read_env`](https://github.com/atto-lang/atto-spec/blob/main/language/file-module-const-v1.md), and [bitwise `<<<`/`>>>`/`ashr` distinct from stream `<<`/`>>`](https://github.com/atto-lang/atto-spec/blob/main/language/bitwise-and-stream-operators-v1.md). These are spec targets, not guaranteed executable features of this legacy repository.

Preserve legacy `.ores` paths, CLI/package/ABI names and wire formats where implemented. No Python support scripts, CI or tests; prefer Rust helper tooling. Document exact compiler SHA/tree, toolchain, executed guest positive/negative checks and unsupported/failed/not-executed outcomes. Host-only or zero-step CI cannot certify feature or backend parity.

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
