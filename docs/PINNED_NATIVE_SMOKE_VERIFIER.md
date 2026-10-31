# Native affine/SGD smoke: pinned guest-output acceptance

**Status: verifier only, not native execution or compiler attestation.**
This gate runs `tests/smoke.ores` only after explicitly supplied,
independently reviewed pins. Its success marker deliberately says
`UNAUTHENTICATED_GUEST_OUTPUT_MATCH`: an arbitrary executable that prints
the expected line can still impersonate a compiler if an operator supplies
its digest. Do not promote to a guest pass without independent provenance of
the compiler binary and the actual source-runner execution record.

## Required inputs

Run from a Git checkout of this ML branch on a source-authorized host with
GNU `sha256sum`, `timeout`, `git`, `cmp` and normal POSIX shell utilities.

- `ORESLANG_BIN`: **absolute path** to the compiler/runner executable.
- `EXPECTED_COMPILER_SHA256`: reviewed immutable SHA-256 for that binary.
- `EXPECTED_ML_HEAD_SHA`: reviewed exact 40-hex HEAD SHA of the ML checkout.
- `EXPECTED_LINEAR_BLOB`: reviewed Git blob SHA of `src/linear.ores`.
- `EXPECTED_SMOKE_BLOB`: reviewed Git blob SHA of `tests/smoke.ores`.

Example, substituting independently established values rather than deriving
approval pins from an untrusted receipt:

```sh
export ORESLANG_BIN=/absolute/path/to/reviewed/oreslang
export EXPECTED_COMPILER_SHA256=<reviewed-64-hex>
export EXPECTED_ML_HEAD_SHA=<reviewed-40-hex>
export EXPECTED_LINEAR_BLOB=<reviewed-40-hex>
export EXPECTED_SMOKE_BLOB=<reviewed-40-hex>
sh scripts/test.sh
```

The verifier returns **78** if missing/unpinned, **1** on failed execution,
unexpected output/stderr or nonzero exit, and **0** only when the pinned
executable exits 0, stderr is empty and stdout equals exactly
`ORES_ML_TRAINING_SMOKE_PASS\n`. The smoke command has a 120-second timeout.
A source SHA and two source blob hashes are recorded; the runner also
rechecks the binary hash after execution. This is source/bytes consistency,
not cryptographic code-signing or an independently authenticated compiler.

## Self-tests

```sh
sh -n scripts/test.sh
sh -n scripts/test-smoke-verifier.sh
sh scripts/test-smoke-verifier.sh
```

The self-test uses a temporary mock Git repository and fake shell programs,
never compiles an Oreslang source file, and proves 11 **structural** cases:
missing compiler, non-absolute path, wrong HEAD, wrong source/fixture blob,
wrong compiler digest, `/bin/true` silent success, stderr pollution, extra
stdout, nonzero exit after correct stdout, and synthetic matched output.

The synthetic-positive case is intentional to show the limit of output
matching: no verifier in this PR authenticates the compiler identity,
Oreslang language semantics, host runner, actor state, or backend parity.
Capture actual source SHA, immutable compiler provenance, compiler command,
runner ID, nonzero steps, stdout/stderr, toolchain, and all required negative
fixtures separately before any real promotion.

Never mirror private source to another organization solely to obtain runner
minutes. Follow the destination allowlist and source authorization policy
before transferring any source tree.
