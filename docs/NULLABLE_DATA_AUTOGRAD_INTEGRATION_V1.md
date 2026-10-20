# P0 CPU nullable data -> reverse-mode autograd integration

**Status: source + isolated host numerical oracle only. NOT verified as Oreslang guest execution.**

The new `examples/train_nullable_data_with_autograd.ores` imports the existing Oreslang-native `FloatColumn` and `compact_supervised_f64_into` from `oreslang-data-science`, and `TapeF64` / `forward_checked_into` / `backward_checked_into` from `oreslang-autograd`. Its inputs include an excluded nullable row; stable original row IDs must be `[0,2,3,4]`. It calculates the squared-error loss and true reverse-mode tape gradients for `w` and `b`, performs 500 full-batch SGD epochs, and requires `w≈2`, `b≈1`, loss below 1e-6. Only `ORES_DATA_AUTOGRAD_TRAINING_PASS` on stdout is accepted by the guest-output checker.

**Three distinct guarantees; do not conflate them:**
1. The data bridge compacts feature/target validity masks into caller-owned mutable buffers, not owned tensor storage.
2. The autograd library performs CPU scalar reverse-mode differentiation; this example does not yet use rank-N Tensor storage, cross-actor gradient routing or the GPU.
3. `train_actor_state_scalar.ores` separately exercises the Oreslang actor's **single** `state {}` field, `with` declared read/write effects, RW loans via `using` and sealed `self.handle_state`, and version-fenced step commits. The combined nullable example itself is sequential, not an actor training pipeline.

In a subsequent real multi-actor program, stage-owned weights, immutable-version leases, optimizer receipts, activation reservations and all saved state live under exactly one `state {}`. A read loan acquires only an immutable pinned parameter snapshot; the loan ends before tensor computation or a send/await. The backward stage checks graph/run/step/epoch/microbatch/edge/version and completes a full fan-in before a short exclusive critical optimizer commit.

The real guest checker `scripts/verify-data-autograd-guest.sh` requires a caller-supplied pinned compiler and sibling source checkouts, fails for missing imports, nonzero process exit and any unexpected output. Host-only CI deliberately does **not** invoke the script. The `tests/nullable_data_autograd_contract.test.mjs` numerical oracle is independent JS code, not an actual compilation of the Oreslang source.

**Release gates:** real exact-commit Java/Graal guest execution of the integrated example; checked source SHA for all 3 repos; Oreslang compilation errors fixed without disabling semantic checks; real LLVM supported-subset parity, actor-owned tensor storage, immutable snapshot lifetime, and multi-stage forward/backward actor execution. Security incidents in `oreslang-platform-conformance` issues 22 and 77 currently block new cross-org private-source mirroring without independent explicit approval. Do not use a private carrier as a shortcut around them.

## Audit: nonfinite math and exact source identity (October 18)

The nullable-to-autograd guest source now checks that the *accumulated* weight gradient, bias gradient, and squared-loss sum remain finite. Every epoch computes separate weight and bias candidates and checks the next loss/weight/bias **before either parameter is assigned**. This closes a partial-update gap in the original demonstration. The numeric policy is not a substitute for tensor-wide checked arithmetic or transactional persistent checkpointing.

The guest acceptance script now requires four caller-supplied pin identities: `ORESLANG_ML_EXPECTED_SHA`, `ORESLANG_DATA_EXPECTED_SHA`, `ORESLANG_AUTOGRAD_EXPECTED_SHA` and `ORESLANG_RUNNER_SHA256`. It verifies the heads of the three independent local checkouts, rejects dirty/untracked files, validates the runner executable's SHA-256 and demands exact stdout. It does not fetch repositories or transfer private source. **The runner executable checksum does not bind dynamically loaded JARs/plugins/libraries or compiler flags**; an actual production provenance receipt must additionally bind those dependencies, the toolchain, source tree and execution environment. The CI script's existence is not guest execution evidence.

Independent host-only structure/numerical regression tests now include candidate overflow, accumulator overflow and rejection without publishing one of two optimizer parameters. Source/org CI may fail before runners execute; those failures cannot be recorded as guest passes. Separate funded source-free mathematical testing in `sagitta-stack/sagitta-infra` is independently authored and explicitly **does not run the Oreslang compiler**. Security incidents #22/#77 continue to prohibit unauthorized source mirroring.
