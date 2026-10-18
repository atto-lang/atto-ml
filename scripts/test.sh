#!/usr/bin/env sh
set -eu
if [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: set ORESLANG_BIN to an executable real Oreslang runner" >&2
  exit 78
fi
"$ORESLANG_BIN" tests/smoke.ores
