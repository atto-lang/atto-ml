#!/usr/bin/env bash
# Actual Oreslang guest acceptance, only in an authorized local workspace.
# Required siblings: oreslang-ml, oreslang-data-science, oreslang-tensor,
# oreslang-autograd. Does not create/fetch/copy any repository.
# Usage: ORESLANG_BIN=/absolute/path/to/oreslang ORESLANG_BIN_SHA256=<64-hex SHA256> bash scripts/verify-supervised-cpu-guest.sh
set -euo pipefail
# Hash the compiler executable itself: six source blobs alone do not
# authenticate the actual runtime used to execute the guest. Do not accept
# relative paths, missing digests, or unsupported hashing tools.
if [[ -z "${ORESLANG_BIN:-}" || ! -f "${ORESLANG_BIN:-}" || ! -x "${ORESLANG_BIN:-}" ]]; then
  echo "NOT_EXECUTED: set ORESLANG_BIN to an existing executable" >&2
  exit 78
fi
if [[ "$ORESLANG_BIN" != /* ]]; then
  echo "NOT_EXECUTED: ORESLANG_BIN must be an absolute path" >&2
  exit 78
fi
if [[ ! "${ORESLANG_BIN_SHA256:-}" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "NOT_EXECUTED: require a 64-digit ORESLANG_BIN_SHA256 digest" >&2
  exit 78
fi
if command -v sha256sum >/dev/null 2>&1; then
  actual_compiler_digest="$(sha256sum -- "$ORESLANG_BIN" | cut -d ' ' -f1)"
elif command -v shasum >/dev/null 2>&1; then
  actual_compiler_digest="$(shasum -a 256 -- "$ORESLANG_BIN" | cut -d ' ' -f1)"
else
  echo "NOT_EXECUTED: SHA-256 verification tool not available" >&2
  exit 78
fi
if [[ "${actual_compiler_digest,,}" != "${ORESLANG_BIN_SHA256,,}" ]]; then
  echo "NOT_EXECUTED: pinned Oreslang executable SHA-256 mismatch" >&2
  exit 78
fi
command -v git >/dev/null || { echo "NOT_EXECUTED: git unavailable" >&2; exit 78; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(dirname "$root")"

require_tree() {
  local repo="$1" expected="$2" base="$workspace/$repo"
  if [[ ! -e "$base/.git" ]]; then
    echo "NOT_EXECUTED: missing dependency repository $repo" >&2
    exit 78
  fi
  local current
  current="$(git -C "$base" rev-parse HEAD)"
  if [[ "$current" != "$expected" ]]; then
    echo "NOT_EXECUTED: dependency HEAD mismatch $repo" >&2
    exit 78
  fi
  # Prevent checkout byte drift or extra untracked import-shadowing files.
  if [[ -n "$(git -C "$base" status --porcelain --untracked-files=all)" ]]; then
    echo "NOT_EXECUTED: dirty dependency checkout $repo" >&2
    exit 78
  fi
}
# Exact reviewed source-repository commit snapshots, not merely six selected
# source files. The per-file blob pins below offer an independent cross-check.
require_tree oreslang-data-science 3d9d273a8fd705c321d60d281bd4ced88afb55b6
require_tree oreslang-tensor e0c2aec1a050f30d06b359f4662adf5f5604bbe6
require_tree oreslang-autograd 455fd9ffb403bb25614103ce6a101cba43e63c2c

require_blob() {
  local repo="$1" path="$2" expected="$3"
  local base="$workspace/$repo"
  if [[ ! -f "$base/$path" || ! -e "$base/.git" ]]; then
    echo "NOT_EXECUTED: missing local source $repo/$path" >&2
    exit 78
  fi
  local actual
  actual="$(git -C "$base" hash-object -- "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "NOT_EXECUTED: dependency blob mismatch $repo/$path expected=$expected actual=$actual" >&2
    exit 78
  fi
}
# Locked, connector-verified immutable Git blob identities. Refresh only
# after reviewing real dependency changes and regenerating tests.
require_blob oreslang-data-science src/data/columnar.ores 8529c60df75d1e9f9d2af98179fd5d02947f712c
require_blob oreslang-data-science src/pipeline/ml_batch_bridge.ores 2d1d5a8173e29f1c2aa432a558f2ba8a912a0f5c
require_blob oreslang-tensor src/tensor.ores 3bf09bf6b98964d15ca145a6c97b74f307bc47b4
require_blob oreslang-autograd src/tape.ores caf8f2a616eb75a0365efa9fb88b4d01cd12b73f
require_blob oreslang-autograd src/forward_checked.ores a14a6410c024581b9e002f858a46fc94cbaeca85
require_blob oreslang-autograd src/backward_checked.ores ea85a8544b89d7047f434a98ff744fd530a83107

# Verify the exact same guest fixture whose behavior was reviewed.
# The integration branch can change independently of library revisions, so
# source file identity must be pinned as well as dependent Git trees.
source="$root/examples/train_supervised_tensor_autograd.ores"
if [[ ! -f "$source" ]]; then
  echo "NOT_EXECUTED: supervised CPU guest fixture is missing" >&2
  exit 78
fi
actual_fixture_blob="$(git -C "$root" hash-object -- "$source")"
if [[ "$actual_fixture_blob" != "0373080c2500df49e476c777c5598463ca126aac" ]]; then
  echo "NOT_EXECUTED: supervised CPU guest fixture Git blob changed" >&2
  exit 78
fi
# A guest that deadlocks or fails to terminate must not monopolize CI
# indefinitely. Require a bounded elapsed wall clock even on manual runners.
if command -v timeout >/dev/null 2>&1; then
  guest_timeout="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
  guest_timeout="$(command -v gtimeout)"
else
  echo "NOT_EXECUTED: a GNU-compatible timeout command is required" >&2
  exit 78
fi
if [[ ! "${ORESLANG_GUEST_TIMEOUT_SECONDS:-120}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NOT_EXECUTED: invalid positive guest timeout seconds" >&2
  exit 78
fi
if (( ${ORESLANG_GUEST_TIMEOUT_SECONDS:-120} > 600 )); then
  echo "NOT_EXECUTED: guest timeout exceeds 600-second ceiling" >&2
  exit 78
fi
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
printf '%s\n' "ORES_ML_SUPERVISED_TENSOR_AUTOGRAD_PASS" > "$temp/expected"
printf 'INVOKING_REAL_ORESLANG_GUEST %s\n' "$source" >&2
if ! "$guest_timeout" -k 5s "${ORESLANG_GUEST_TIMEOUT_SECONDS:-120}s" "$ORESLANG_BIN" "$source" > "$temp/actual" 2> "$temp/stderr"; then
  cat "$temp/stderr" >&2
  echo "ORESLANG_GUEST=FAILED; no pass receipt" >&2
  exit 1
fi
if ! cmp -s "$temp/expected" "$temp/actual"; then
  diff -u "$temp/expected" "$temp/actual" >&2 || true
  cat "$temp/stderr" >&2
  echo "ORESLANG_GUEST=OUTPUT_MISMATCH; no pass receipt" >&2
  exit 1
fi
echo "ORESLANG_GUEST=PASS; COMPILER_SHA256=$actual_compiler_digest; DEPENDENCY_TREES=3; DEPENDENCY_BLOBS=6; FIXTURE_BLOB=1; CPU_AUTOGRAD=VERIFIED"
