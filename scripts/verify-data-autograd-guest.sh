#!/usr/bin/env bash
# Real multi-repository Oreslang guest gate, NOT a source-free oracle.
# Requires checked-out sibling oreslang-data-science and oreslang-autograd
# with separately reviewed/pinned exact source identities.
set -euo pipefail
if (($# == 0)); then
  echo "usage: $0 PINNED_ORESLANG_RUNNER [RUNNER_ARGS...]" >&2
  exit 64
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(dirname "$root")"
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
