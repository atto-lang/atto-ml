# atto-ml

Machine-learning primitives and model execution infrastructure for AttoLang.

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

AttoLang is the public name; preserve existing Oreslang/.ores compatibility and use canonical .atto for new portable library source. This README defines scope, not feature readiness.

**Acceptance:** name exact source SHA/tree, runtime/toolchain, positive and negative checks and execution receipt before claiming support. Host-only tests and empty CI steps do not prove guest execution.
