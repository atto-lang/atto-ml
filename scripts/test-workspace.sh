#!/usr/bin/env sh
set -eu
if [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: a real executable ORESLANG_BIN is required" >&2
  exit 78
fi
if [ ! -f "../atto-autograd/src/tape.ores" ]; then
  echo "UNCONFIGURED: clone atto-autograd beside atto-ml" >&2
  exit 78
fi
if [ ! -f "../atto-tensor/src/row_batches.ores" ]; then
  echo "UNCONFIGURED: clone atto-tensor row-batch branch beside atto-ml" >&2
  exit 78
fi

tmpdir=$(mktemp -d) || exit 78
trap 'rm -rf "$tmpdir"' EXIT

run_case() {
  fixture="$1"
  marker="$2"
  if ! "$ORESLANG_BIN" "$fixture" >"$tmpdir/stdout" 2>"$tmpdir/stderr"; then
    echo "GUEST_FAILED: $fixture" >&2
    cat "$tmpdir/stderr" >&2
    exit 1
  fi
  printf '%s\n' "$marker" >"$tmpdir/expected"
  if [ -s "$tmpdir/stderr" ] || ! cmp -s "$tmpdir/expected" "$tmpdir/stdout"; then
    echo "GUEST_OUTPUT_MISMATCH: $fixture" >&2
    cat "$tmpdir/stderr" >&2
    exit 1
  fi
  printf '%s\n' "$marker"
}

run_case examples/train_with_autograd.ores ORES_ML_AUTOGRAD_TRAINING_PASS
run_case examples/train_two_layer_autograd.ores ORES_ML_TWO_LAYER_RELU_AUTOGRAD_PASS
run_case examples/train_aligned_dataset.ores ORES_ML_ALIGNED_DATASET_TRAINING_PASS
run_case examples/train_adamw_smoke.ores ORES_ML_ADAMW_CHECKED_PASS
run_case examples/train_tensor_mlp_vjp_cpu.ores ORES_ML_TENSOR_TWO_LAYER_VJP_PASS
