#!/usr/bin/env sh
set -eu
if [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: a real executable ORESLANG_BIN is required" >&2
  exit 78
fi
if [ ! -f "../oreslang-autograd/src/tape.ores" ]; then
  echo "UNCONFIGURED: clone oreslang-autograd beside oreslang-ml" >&2
  exit 78
fi
"$ORESLANG_BIN" examples/train_with_autograd.ores
"$ORESLANG_BIN" examples/train_two_layer_autograd.ores
if [ ! -f "../oreslang-tensor/src/row_batches.ores" ]; then
  echo "UNCONFIGURED: clone oreslang-tensor row-batch branch beside oreslang-ml" >&2
  exit 78
fi
"$ORESLANG_BIN" examples/train_aligned_dataset.ores
