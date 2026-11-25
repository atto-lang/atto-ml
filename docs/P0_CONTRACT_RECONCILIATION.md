# P0 ML implementation and v1 contract reconciliation

**Draft, source-only. No native Oreslang guest execution has been certified.**

## Why this branch exists

PR #1's implementation branch diverged from `main` and conflicts with the
default branch's `docs/CONTRACT.md`. Its former contract document describes
the P0 affine/SGD reference, while current `main`'s contract defines
future `integration/contract.v1.json` interfaces and truthfully marks them
`PROPOSED_CONTRACT_NOT_IMPLEMENTATION`.

The reconciliation branch is based on main commit
`a90feeae18a89d6b2f76ddc9390d473823a86b62` and imports the
**exact final file blobs** of reviewed draft ML PR #29 head
`da4cfc35ad3262b3067d7e328fa2e794d5714a20`, stacked on
PR #26 (`30b4a8d330875c4061435f4cc4ead6fd`) and PR #1
(`36c3a941080b47592445dd8a76ae2f2115c8d658`).

All transfers are between Git objects in **the same private repository**.
No source is transferred across GitHub organizations.

## Explicit conflict resolution and preservation

- Keep **current main** `docs/CONTRACT.md` unchanged, as the canonical
  prospective v1 interface contract. `integration/contract.v1.json`,
  `test/contract.test.mjs`, `README.md`, `package.json`, and
  `.github/workflows/contract.yml` are byte-identical across both trees.
- Preserve PR #1's different `docs/CONTRACT.md` **verbatim** under
  `docs/P0_IMPLEMENTATION_DRAFT.md`, so no implementation claims are lost
  and they cannot silently replace the release contract.
- Retain all main-only files without modification, especially
  `.github/PULL_REQUEST_TEMPLATE.md`, `.github/workflows/rollout-evidence.yml`,
  and the full `tools/rollout-evidence/` Rust receipt-inspection subtree.
- Copy the draft head's ML `.ores` sources, examples, numerical oracles,
  workflows, scripts, smoke fixture and output golden using their **existing,
  immutable Git blob IDs** rather than inventing or regenerating implementation
  content.
- The PR #26 SGD alias and empty-vector checks are included. The stacked
  PR #29 independently pinned source/tree/guest-output verifier and its
  **20** source-free negative tests are included. The independent affine/SGD
  host numerical oracle and expanded `.ores` guest assertions are included.
- PRs #25 (stricter Rust receipt input) and #28 (contract regression pinning)
  remain independent review work. This branch does not silently cherry-pick,
  merge or claim execution of either.

## Source-only checks versus executable evidence

Available host-only verification checks source blobs and fixture shape and
can run `npm test`, `node --test tests/*.test.mjs` and
`sh scripts/test-smoke-verifier.sh` on an authorized clone. Host-side
numerical oracles are **not** an Oreslang compiler, actor runtime, native
Java/GraalVM, LLVM AOT or five-backend parity test.

Real guest execution must use reviewed SHA pins for the source HEAD, root
tree, source/fixture blobs and an independently authenticated compiler binary,
then record the exact command, stdout/stderr, real runner/job steps,
positive convergence and negative state-preservation outcomes.
`scripts/test.sh` emits `UNAUTHENTICATED_GUEST_OUTPUT_MATCH` only;
its matching of arbitrary process output is not proof of compiler identity.

At reconciliation time GitHub Actions in the source organization still fails
before any runner steps; see issue #20. Do not interpret a failed zero-step
run as a failing implementation or passing native test. The newly reconciled
source MUST remain **draft** until administrator-visible CI admission and
actual pinned Java/GraalVM + LLVM guest execution succeed.

No automatic merge/release, branch protection bypass, or cross-organization
private source export is authorized.

## AttoLang integration checkpoint

The preceding historical account describes the original legacy reconciliation, not the current AttoLang tree. AttoLang PR #21 now incorporates current main `e143a7b5711238c97e015c6c5b632e09612b5a4f`, retaining its README, license, capability inventory, tests and all governance files. The 24 P0 paths are carried forward from AttoLang head `4be7e3244bcbbc83c6a7362e0ff01ca2902f16c0`; this branch includes migration-era edits, so it must not be treated as byte-identical to legacy PR #29 or current AttoLang PR #20. Compare their overlapping source and tests before treating them as superseded. The current repository is public; no private dependency or compiler source was exported. Native guest release gates remain unchanged and unexecuted.
