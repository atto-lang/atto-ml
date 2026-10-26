# KAN CPU native acceptance: source-local, exact-head, three-fixture gate

**Status: NOT_EXECUTED until a real pinned compiler and an authorized runner execute these fixtures.** Host mathematical tests, CI shell syntax and mock receipts must never be advertised as guest execution.

## Current blocker

[CI runner admission issue #20](https://github.com/ores-truffle-oreslang/oreslang-ml/issues/20) tracks Actions failures with `steps=null`, even for metadata-only jobs. This is a scheduler/infrastructure *symptom*; it does **not** establish exhausted minutes or any native numerical failure.

Private source must remain in `ores-truffle-oreslang/oreslang-ml` (numeric GitHub repo ID `1423913534`) absent independent, exact destination authorization. The Java source-governance allowlist contains no approved destination. A copied Git tree in another organization, even through the GitHub connector without tokens, constitutes private-source transfer. GitHub Actions minutes do not grant authorization.

## Required reviewed inputs

- A clean source checkout of the **reviewed exact PR head**. `EXPECTED_SOURCE_COMMIT` must be set from an independent trusted review record, not calculated from the untrusted worktree being tested.
- A trusted native Oreslang compiler/executable at absolute `ORESLANG_BIN`. Its independent SHA-256 pin must be supplied via `ORESLANG_BIN_SHA256`; a digest selected by the PR author or the code under test is not an attestation.
- Runner access to the three inherited native guest fixture scripts, their prechecked and pinned source/fixture Git blobs, and GNU-compatible `timeout`. The verifier requires an unchanged clean repository checkout.
- Java/GraalVM **and separately** LLVM toolchains must each run the same three-fixture matrix. Runner availability alone is not evidence of runtime parity.

## Source-local execution

From a clean and authorized checkout, after **independently** setting all three required variables:

```bash
# Variables must be injected by a trusted administrator or reviewed CI gate:
#   EXPECTED_SOURCE_COMMIT: exact 40-character approved source PR head
#   ORESLANG_BIN: absolute executable compiler path
#   ORESLANG_BIN_SHA256: independently approved compiler SHA256
bash scripts/verify-kan-native-cpu-suite.sh
```

The suite invokes, strictly in order:

1. `verify-kan-multispan-guest.sh`: clamped cubic forward, gradients, rejection and right-hand knot semantics.
2. `verify-kan-layer-guest.sh`: packed KAN layer forward, alias and failure-atomic publication.
3. `verify-kan-layer-vjp-guest.sh`: packed input/parameter/bias VJP, aliases, inactive derivative span, negative cases and output atomicity.

Each child must produce exactly one **byte-for-byte approved** receipt containing `GUEST=PASS`, the exact source and compiler pins, and the independently pinned kernel and fixture Git blob hashes and must independently validate its pinned kernel and fixture blobs, source HEAD, clean worktree, SHA256, stderr and actual guest stdout. Any failed child, unexpected stderr, missing/altered hash, additional field, malformed receipt or extra output line prevents the suite-level `KAN_CPU_SUITE=PASS; GUEST_FIXTURES=3` line.

**Important:** The suite's receipt parser is not a compiler identity authority. `scripts/test-kan-native-suite-receipts.sh` uses synthetic child scripts and proves only parser behavior. `scripts/test-kan-guest-ci-identity.sh` validates CI environment rejection logic and likewise does not execute Oreslang.

## CI evidence grades

- **Runner-admission failure:** `steps=null`; neither suite execution nor code failure.
- **Host reference:** `node --test` completed with nonempty runner steps; proves only independently checked JavaScript arithmetic.
- **Mock/verifier gate:** shell syntax and synthetic receipt checks; no compiler proof.
- **Native guest accepted:** all three source- and compiler-pinned Oreslang guest fixtures run on an authorized runner, with exact stdout/stderr and real nonempty job steps; separately for Java/GraalVM and LLVM.
- **Training accepted:** in addition to native execution, owned `NdArray` and actor-state version-fenced optimizer implementation + held-out nonlinear training tests pass. This milestone has not been reached.

Keep [PR #19](https://github.com/ores-truffle-oreslang/oreslang-ml/pull/19) draft until actual native acceptance, and do not merge stacked dependent PRs on the basis of host-only tests.

## Latest hardening — exact native receipts

The native suite aggregator previously admitted any receipt sharing a source/compiler prefix, even with missing or incorrect guest kernel/fixture Git blob hashes. It now owns separate immutable, complete receipt expectations for the spline, forward layer and VJP, and compares each child stdout byte-for-byte including its terminating newline, while rejecting any successful child's stderr. This is an independent consistency guard, **not** authentication of an arbitrary caller-supplied compiler binary. The source-free mock suite exercises 20 scenarios, including successful matching receipts, incorrect kernel or fixture blobs, omitted hashes, added metadata, stderr, duplicate blank lines and error returns. Passing mocks never proves native Oreslang execution.

A source-free alternate-org [admission canary #58](https://github.com/ores-dummy-org-oreslang-stack/ores-dummy-org-oreslang-stack-infra/pull/58) also failed before any runner steps ([run #40253849945](https://github.com/ores-dummy-org-oreslang-stack/ores-dummy-org-oreslang-stack-infra/actions/runs/40253849945)). This does not establish billing exhaustion or private-source-transfer authority; keep source in the approved source organization. See [runner issue #20](https://github.com/ores-truffle-oreslang/oreslang-ml/issues/20).


## Aggregate preflight defense in depth — 2026-10-26

The aggregate `scripts/verify-kan-native-cpu-suite.sh` now **independently** checks the exact clean Git `HEAD` against externally supplied `EXPECTED_SOURCE_COMMIT`, the existence and executability of the *absolute* `ORESLANG_BIN`, and its real SHA-256 bytes against `ORESLANG_BIN_SHA256`. It also rejects suspicious CI repository identifiers before launching any fixture. The individual guest verifiers retain their separate identity, blob, timeout and compiler-hash checks.

These controls prevent a counterfeit child-script pass receipt from bypassing source/compiler preflight by supplying only a plausible-looking string. They do not establish the trustworthiness of a compiler if the expected hash itself was supplied by untrusted PR code; hashes and the runner must be approved through a separate trusted channel.

The source-free `scripts/test-kan-native-suite-receipts.sh` now creates a temporary, **real committed Git repository**, uses an external executable containing only `exit 11` whose bytes are genuinely SHA-256 hashed, and stubs all three native verifiers to test the aggregator's failure and receipt logic. The dummy compiler is never executed as a guest compiler. **20/20 local test cases passed**; all mocked success cases remain **NATIVE_GUEST=NOT_EXECUTED**.

Acceptance still requires a secure, source-authorized runner executing actual Java/GraalVM and LLVM compilers independently, with reviewed compiler hashes and source commits. No native pass is claimed.
