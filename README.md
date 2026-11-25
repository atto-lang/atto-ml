> **AI agent instructions (organization-wide):** Before working on code, issues, or pull requests, read and follow the [organization `AGENTS.md`](https://github.com/atto-lang/.github/blob/main/AGENTS.md), including its **merge-first stealth-development policy**. Honor any repository-local `AGENTS.md`, explicit holds, and GitHub-enforced rules.

# atto-ml

Machine-learning primitives and model execution infrastructure for AttoLang.

## Naming migration and language authority

This repository is maintained at [atto-lang/atto-ml](https://github.com/atto-lang/atto-ml); use `https://github.com/atto-lang/atto-ml.git` for Git remotes and canonical `.atto` source examples. The normative language contract is [atto-spec](https://github.com/atto-lang/atto-spec), with executable guest evidence tracked by [atto-platform-conformance](https://github.com/atto-lang/atto-platform-conformance). Follow current ownership, borrowing, effects, actor, and syntax rules there; do not infer implemented features from a README or unexecuted CI.

## Scope

Machine-learning APIs, model/tensor operations, and backend-specific execution adapters.

## API and portability principles

- Specify numerical/semantic behavior before assuming support on every backend.
- Document input types, error behavior, shape and memory ownership requirements as applicable.
- Distinguish CPU execution, SIMD, GPU offload, and other target-specific capabilities.

## CPU tensor training (candidate)

**Native guest acceptance remains unverified.** The API and behavior below describe the candidate source, not a certified backend implementation.

`Dense.forward_f64_into` computes a row-major affine layer using the Tensor
matrix kernel. `Mlp` implements Dense → ReLU → Dense for f64 batches shaped
`[rows, features]`. Initialize the model with parameters in `[W1, b1, W2, b2]`
order and an SGD or AdamW configuration. `Mlp.predict` returns a tensor shaped
`[rows, outputs]`. `Mlp.grad` returns mean squared error and writes the
parameter gradient in the same packed order.

`Mlp.fit` runs full-batch steps under an exclusive model borrow. It publishes
parameters, optimizer moments, and the step counter together after all steps
and final loss validation succeed. `Mlp.checkpoint` and `Mlp.restore` copy
these values into and out of a versioned in-memory snapshot. Restore creates
an independent model and preserves the optimizer's next step. The CPU profile
bounds layer widths, buffer lengths, and the work per fit.

Run `scripts/test-workspace.sh` with `ORESLANG_BIN` and matching sibling
checkouts of `atto-tensor` and `atto-autograd` to execute the training and
checkpoint fixtures.

## Verification

Use checked-in tests, documented tolerances, and exact-commit runtime evidence. Host-only or compilation tests are not equivalent to executed AttoLang guest tests. Record backend gaps explicitly.

## Development status

This describes the repository's intended purpose, not a verified list of implemented features. Refer to checked-in code and CI for current functionality and validated commands.

## Related

- [AttoLang](https://github.com/atto-lang)
- [ML conformance](https://github.com/atto-lang/atto-ml-conformance)
- [Language specification](https://github.com/atto-lang/atto-spec)

## Repository ownership and contribution routing

**Belongs here:** High-level model execution APIs, training/inference orchestration and ML backend adapters.

**Belongs elsewhere:** Tensor layouts/math in atto-tensor, autodiff in atto-autograd, numerical conformance in atto-ml-conformance.

**Security and portability boundary:** Document memory budgets, trusted model formats, deterministic seeds, numerical tolerance and CPU/GPU capability fallbacks without inventing backend parity.

AttoLang is the canonical project name; new source, examples and documentation use `.atto` and the `atto-lang` repository organization. This README defines scope, not feature readiness.

**Acceptance:** name exact source SHA/tree, runtime/toolchain, positive and negative checks and execution receipt before claiming support. Host-only tests and empty CI steps do not prove guest execution.

## Organization boundaries and legacy compatibility

Portable ML algorithms, model-side APIs and language tooling belong in the AttoLang ecosystem; commercial hosted model serving, deployment, registries, tenant identity and metered billing belong in separately governed cloud services, not this library. Keep legacy `.ores` examples and historical package/CLI aliases compatible until their `.atto` replacements and consumers have been compiler-validated. Similar repository names are not evidence of identity-preserving transfer or source parity.

Any future repository transfer or commercial spinout requires a reviewed owner/rollback plan, independently authorized source and destination, recorded repository IDs and Git commit/tree, verified access/protection, package consumers, webhooks, Actions and release continuity. Never transfer private source without explicit authorization. Verify history, controls and stepful exact-revision CI after any move; do not treat missing runner steps or host-only tests as compiled guest evidence.
