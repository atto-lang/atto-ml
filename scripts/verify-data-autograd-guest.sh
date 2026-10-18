#!/usr/bin/env bash
# Explicit exact-source guest gate for the CPU nullable-data/autograd example.
# This reads sibling source repos locally; NO cross-organization mirror.
# Requires a reviewed compiler/runner and a SHA-256 for its executable.
set -euo pipefail
if (($# == 0)); then
  echo "usage: $0 ABSOLUTE_PINNED_ORESLANG_RUNNER [RUNNER_ARGS...]" >&2
  exit 64
fi
for required in ORESLANG_ML_EXPECTED_SHA ORESLANG_DATA_EXPECTED_SHA \
  ORESLANG_AUTOGRAD_EXPECTED_SHA ORESLANG_RUNNER_SHA256; do
  if [[ -z "${!required:-}" ]]; then
    echo "missing required source/binary identity: $required" >&2
    exit 64
  fi
done
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(dirname "$root")"
verify_repo() {
  local dir="$1" expected="$2" actual
  if [[ ! -d "$dir/.git" ]]; then
    echo "expected independent Git checkout: $dir" >&2
    exit 78
  fi
  actual="$(git -C "$dir" rev-parse HEAD)"
  if [[ "$actual" != "$expected" ]]; then
    echo "source revision mismatch for $dir (expected $expected; got $actual)" >&2
    exit 78
  fi
  if [[ -n "$(git -C "$dir" status --porcelain --untracked-files=all)" ]]; then
    echo "dirty source worktree is not approved: $dir" >&2
    exit 78
  fi
}
verify_repo "$root" "$ORESLANG_ML_EXPECTED_SHA"
verify_repo "$workspace/oreslang-data-science" "$ORESLANG_DATA_EXPECTED_SHA"
verify_repo "$workspace/oreslang-autograd" "$ORESLANG_AUTOGRAD_EXPECTED_SHA"
runner="$1"
if [[ "$runner" != /* || ! -f "$runner" || ! -x "$runner" ]]; then
  echo "runner must be an absolute executable file path" >&2
  exit 64
fi
if command -v sha256sum >/dev/null 2>&1; then
  runner_hash="$(sha256sum "$runner")"; runner_hash="${runner_hash%% *}"
elif command -v shasum >/dev/null 2>&1; then
  runner_hash="$(shasum -a 256 "$runner")"; runner_hash="${runner_hash%% *}"
else
  echo "no SHA-256 file digest executable available" >&2
  exit 78
fi
if [[ "$runner_hash" != "$ORESLANG_RUNNER_SHA256" ]]; then
  echo "runner executable hash mismatch" >&2
  exit 78
fi
for p in "$workspace/oreslang-data-science/src/pipeline/ml_batch_bridge.ores" \
         "$workspace/oreslang-autograd/src/tape.ores" \
         "$workspace/oreslang-autograd/src/forward_checked.ores" \
         "$workspace/oreslang-autograd/src/backward_checked.ores"; do
  if [[ ! -f "$p" ]]; then
    echo "missing sibling Oreslang source: $p" >&2
    exit 78
  fi
done
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
printf 'ORES_DATA_AUTOGRAD_TRAINING_PASS\n' > "$temp/expected"
if ! "$@" "$root/examples/train_nullable_data_with_autograd.ores" > "$temp/stdout" 2> "$temp/stderr"; then
  cat "$temp/stderr" >&2
  echo "ORESLANG_DATA_AUTOGRAD_GUEST=FAILED" >&2
  exit 1
fi
if ! diff -u "$temp/expected" "$temp/stdout"; then
  cat "$temp/stderr" >&2
  echo "ORESLANG_DATA_AUTOGRAD_GUEST=OUTPUT_MISMATCH" >&2
  exit 1
fi
echo "ORESLANG_DATA_AUTOGRAD_GUEST=PASS"
