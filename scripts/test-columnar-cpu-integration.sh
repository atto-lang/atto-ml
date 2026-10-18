#!/bin/sh
# REAL guest integration, never a JS fallback. Layout:
# parent/{oreslang-ml,oreslang-data-science,oreslang-tensor,oreslang-autograd}
# Usage: ORESLANG_BIN=/absolute/path/to/pinned/oreslang sh scripts/test-columnar-cpu-integration.sh
# Pin the exact dependency bytes, avoiding accidental execution against stale
# or partially checked-out compiler/library source.
set -eu
if [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "${ORESLANG_BIN:-}" ]; then
  echo "NOT_EXECUTED: set ORESLANG_BIN to an executable real Oreslang runner" >&2
  exit 78
fi
case "$ORESLANG_BIN" in
  /*) ;;
  *) echo "NOT_EXECUTED: ORESLANG_BIN must be an absolute path" >&2; exit 78 ;;
esac
if ! command -v git >/dev/null 2>&1; then
  echo "NOT_EXECUTED: git is required for exact file-blob checks" >&2
  exit 78
fi
root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
parent="$(dirname "$root")"
pin_file() {
  file="$1"
  expected="$2"
  if [ ! -f "$file" ]; then
    echo "NOT_EXECUTED: missing required source $file" >&2
    exit 78
  fi
  actual="$(git hash-object "$file")"
  if [ "$actual" != "$expected" ]; then
    echo "NOT_EXECUTED: dependency blob mismatch: $file" >&2
    echo "expected=$expected actual=$actual" >&2
    exit 78
  fi
  printf 'SOURCE_BLOB_PASS %s %s\n' "$expected" "$file"
}
pin_file "$parent/oreslang-data-science/src/pipeline/ml_batch_bridge.ores" 2d1d5a8173e29f1c2aa432a558f2ba8a912a0f5c
pin_file "$parent/oreslang-data-science/src/data/columnar.ores" 8529c60df75d1e9f9d2af98179fd5d02947f712c
pin_file "$parent/oreslang-tensor/src/tensor.ores" 3bf09bf6b98964d15ca145a6c97b74f307bc47b4
pin_file "$parent/oreslang-tensor/src/row_batches.ores" 36a2f5cc4fca565c8927160cfac7af9561004190
pin_file "$parent/oreslang-tensor/src/matmul_vjp.ores" 11a2ee1a49b4e1352cd7b0171d8ada9ab1b4ea31
pin_file "$parent/oreslang-autograd/src/tape.ores" caf8f2a616eb75a0365efa9fb88b4d01cd12b73f
pin_file "$parent/oreslang-autograd/src/forward_checked.ores" a14a6410c024581b9e002f858a46fc94cbaeca85
pin_file "$parent/oreslang-autograd/src/backward_checked.ores" ea85a8544b89d7047f434a98ff744fd530a83107
pin_file "$parent/oreslang-autograd/src/row_gather_vjp.ores" 7a92ab3b4f09832129fa738f68e07fb566064d65

test_fixture="$root/examples/train_columnar_tensor_autograd_cpu.ores"
expected_stdout="$root/tests/golden/columnar_tensor_autograd_cpu.stdout"
if [ ! -f "$test_fixture" ] || [ ! -f "$expected_stdout" ]; then
  echo "NOT_EXECUTED: fixture or expected output missing" >&2
  exit 78
fi
printf 'FIXTURE_BLOB %s\n' "$(git hash-object "$test_fixture")"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
if ! (cd "$root" && "$ORESLANG_BIN" "$test_fixture") >"$tmp/actual" 2>"$tmp/stderr"; then
  cat "$tmp/stderr" >&2
  echo "ORESLANG_GUEST_EXECUTION=FAILED" >&2
  exit 1
fi
if ! cmp -s "$expected_stdout" "$tmp/actual"; then
  diff -u "$expected_stdout" "$tmp/actual" >&2 || true
  cat "$tmp/stderr" >&2
  echo "ORESLANG_GUEST_OUTPUT=FAILED" >&2
  exit 1
fi
echo "ORESLANG_COLUMNAR_TENSOR_AUTOGRAD_GUEST_PASS"
