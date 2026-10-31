# Oreslang ML: audit and release gates (2026-10-30)

**Status: no native ML implementation on `main` is certified.** The default branch
contains the proposed v1 integration contract, metadata tests and an offline
receipt *consistency* inspector. Most native ML prototypes remain unmerged,
stacked **draft** pull requests. Neither host numerical tests nor a populated
receipt TSV can certify a compiled Oreslang guest.

## Inspected scope and unresolved P0 gates

| Area | Source/evidence | Required gate before promotion |
| --- | --- | --- |
| Actor control and replay | [issue #11](../issues/11), [draft PR #24](../pull/24) | Sealed command variants; runtime-attested supervisor CONTROL/capability origin; non-reused actor incarnation/run IDs; reject unauthorized init, close, optimizer commit and duplicate/stale gradient mail, including restart ABA. |
| Tensor MLP and AdamW | [draft PR #23](../pull/23), [issue #2](../issues/2) | Exact compiler-compiled Tensor/Autograd integrations, native finite-difference gradient checks, all-or-nothing owned parameter/moment publication, deterministic decreasing loss and checked rollback on every invalid numeric input. |
| KAN VJP and finite numerical bounds | [draft PR #19](../pull/19) stacked on #17/#15/#14 | Exact-head guest compilation; independent numerical oracles including repeated knots, active spans, overflow, aliased writable buffers, fan-in and late-output rejection. |
| Runner admission | [issue #20](../issues/20) | Source-authorized Actions job with nonempty executed steps and exact source SHA; separate real Java/GraalVM and LLVM guest receipts. Zero-step failure is **NOT_EXECUTED**, not a source pass or test failure. |
| Default-branch governance | `main` inspected 2026-10-30 (unprotected at inspection) | Enforce review and required real checks before merging; do not count a metadata-only workflow or offline receipt check as native acceptance. |

The acceptance contract's `mainSnapshotSha` is a **historical bootstrap
snapshot**, not an assertion that the current `main` tree or any pending PR
was executed. Contract v1 IDs, fixture provenance and evidence status are now
explicitly regression-pinned; incompatible contract changes require a version
migration and a corresponding reviewed consumer update.

## Mandatory technical acceptance

1. **Ownership and cancellation:** use actor-owned storage and generation-safe
   read-only loans, with all source tasks and scoped descendants physically
   quiescent before arena teardown. Reject use-after-free, aliasing of writable
   arrays, unsafe long-lived borrow across `await`, and partial publication
   when cancellation races with optimizer commit.
2. **Numerical integrity:** preflight shapes, quotas, dtype/device, IEEE-754
   nonfinites, ill-conditioned gradients, intermediate overflow, and
   bias-correction degeneracy. Keep public weights, moments and gradient
   destinations unchanged on rejected steps; scratch may be expendable only
   when explicitly documented. Compare independent finite-difference VJPs
   and deterministic convergence fixtures with declared tolerances.
3. **Control-plane integrity:** a payload's caller-supplied run/step/version
   is not an authentication credential. Capability, actor incarnation,
   expected version, fan-in completeness, and proposal validity must be
   checked in one exclusive commit transaction; replay and restart must
   never reapply a weight update.
4. **Bounded execution:** fixed allocator/arena ownership and explicit
   maximum tensor shapes, memory, mailbox counts, cancellation cleanup and
   scheduling budgets. LLVM zero-GC and hard-real-time capability remain
   *targets*, not claims, without worst-case execution evidence.
5. **Backend parity:** record source SHA, root tree, compiler binary SHA-256,
   dependency SHAs, fixture blob IDs, exact command, toolchain/target,
   nonempty runner steps, exit code, stdout/stderr, numerical oracle and
   negative diagnostics separately for Java/GraalVM, LLVM, Wasm, .NET and
   Forge. Unsupported cases must report explicit `NOT_EXECUTED` or
   unsupported status, not a synthetic pass.

## Private-source and receipt handling

Do not transfer a private Git tree into another GitHub organization solely
because it has available Actions minutes. A temporary branch is still a
private-source export. Require independent destination and exact-tree
authorization under [Java issue #1501](https://github.com/ores-truffle-oreslang/oreslang-source.java/issues/1501)
and the private-CI allowlist. No such approval is established by this audit.

The Rust receipt checker rejects malformed, noncanonical or incomplete
metadata and emits `CONSISTENT_UNAUTHENTICATED_RECEIPT` for structurally
consistent rows. It does not contact GitHub, authenticate job provenance,
hash files, run a compiler, or elevate any draft. Canonical input must end
in LF and contain only printable ASCII, tab delimiters and LF line breaks;
NUL, DEL, terminal escape, CRLF and an unterminated row are rejected.
Independently verify each purported run and artifact before any promotion.

**Merge policy:** keep ML feature PRs draft until their specific native
gates execute; this document and the metadata tests are not a substitute.
