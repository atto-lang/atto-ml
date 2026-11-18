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
