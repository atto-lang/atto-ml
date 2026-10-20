#!/usr/bin/env bash
# Real two-repository native Oreslang guest gate for an actor-owned
# two-layer ReLU/autograd training step with state-version CAS. Never fetches or
# publishes private source and never treats a JS oracle as compiler evidence.
set -euo pipefail
if (($# == 0)); then
  echo "usage: $0 ABSOLUTE_PINNED_ORESLANG_RUNNER [RUNNER_ARGS...]" >&2
  exit 64
fi
for required in ORESLANG_ML_EXPECTED_SHA ORESLANG_AUTOGRAD_EXPECTED_SHA \
  ORESLANG_RUNNER_SHA256; do
  if [[ -z "${!required:-}" ]]; then
    echo "missing required pinned source/runner identity: $required" >&2
    exit 64
  fi
done
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(dirname "$root")"
verify_repo() {
  local dir="$1" expected="$2" actual
  if [[ ! -e "$dir/.git" ]]; then
    echo "missing independent Git worktree: $dir" >&2
    exit 78
  fi
  actual="$(git -C "$dir" rev-parse HEAD)"
  if [[ "$actual" != "$expected" ]]; then
    echo "Git source revision mismatch for $dir" >&2
    exit 78
  fi
  if [[ -n "$(git -C "$dir" status --porcelain --untracked-files=all)" ]]; then
    echo "dirty or untracked source worktree: $dir" >&2
    exit 78
  fi
}
verify_repo "$root" "$ORESLANG_ML_EXPECTED_SHA"
verify_repo "$workspace/oreslang-autograd" "$ORESLANG_AUTOGRAD_EXPECTED_SHA"
runner="$1"
if [[ "$runner" != /* || ! -f "$runner" || ! -x "$runner" ]]; then
  echo "runner must be an absolute executable file" >&2
  exit 64
fi
if command -v sha256sum >/dev/null 2>&1; then
  runner_hash="$(sha256sum "$runner")"; runner_hash="${runner_hash%% *}"
elif command -v shasum >/dev/null 2>&1; then
  runner_hash="$(shasum -a 256 "$runner")"; runner_hash="${runner_hash%% *}"
else
  echo "SHA-256 digest utility unavailable" >&2
  exit 78
fi
if [[ "$runner_hash" != "$ORESLANG_RUNNER_SHA256" ]]; then
  echo "Oreslang guest runner executable hash mismatch" >&2
  exit 78
fi
for source in \
  "$workspace/oreslang-autograd/src/tape.ores" \
  "$workspace/oreslang-autograd/src/forward_checked.ores" \
  "$workspace/oreslang-autograd/src/backward_checked.ores"; do
  if [[ ! -f "$source" ]]; then
    echo "required pinned sibling Oreslang source missing: $source" >&2
    exit 78
  fi
done
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
printf 'ORES_ACTOR_TWO_LAYER_AUTOGRAD_PASS\n' > "$temp/expected"
if ! "$@" "$root/examples/train_actor_two_layer_autograd.ores" \
    > "$temp/stdout" 2> "$temp/stderr"; then
  cat "$temp/stderr" >&2
  echo "ORESLANG_ACTOR_TWO_LAYER_GUEST=FAILED" >&2
  exit 1
fi
if ! diff -u "$temp/expected" "$temp/stdout"; then
  cat "$temp/stderr" >&2
  echo "ORESLANG_ACTOR_TWO_LAYER_GUEST=OUTPUT_MISMATCH" >&2
  exit 1
fi
echo "ORESLANG_ACTOR_TWO_LAYER_GUEST=PASS"
