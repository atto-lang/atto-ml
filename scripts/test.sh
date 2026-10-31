#!/usr/bin/env sh
# Strict, source-pinned .ores guest-output admission. Not compiler attestation.
set -eu

unconfigured() {
  printf 'NOT_EXECUTED: %s\n' "$1" >&2
  exit 78
}

valid_hex() {
  value="$1"
  size="$2"
  [ "${#value}" -eq "$size" ] || return 1
  case "$value" in
    '' | *[!0123456789abcdef]*) return 1 ;;
  esac
  [ -n "$(printf '%s' "$value" | tr -d '0')" ] || return 1
}

[ -n "${ORESLANG_BIN:-}" ] || unconfigured 'set ORESLANG_BIN to a pinned real compiler executable'
case "$ORESLANG_BIN" in
  /*) ;;
  *) unconfigured 'ORESLANG_BIN must be an absolute path' ;;
esac
[ -x "$ORESLANG_BIN" ] || unconfigured 'ORESLANG_BIN is not executable'

valid_hex "${EXPECTED_ML_HEAD_SHA:-}" 40 || unconfigured 'EXPECTED_ML_HEAD_SHA must be a nonzero 40-hex commit'
valid_hex "${EXPECTED_LINEAR_BLOB:-}" 40 || unconfigured 'EXPECTED_LINEAR_BLOB must be a nonzero 40-hex Git blob'
valid_hex "${EXPECTED_SMOKE_BLOB:-}" 40 || unconfigured 'EXPECTED_SMOKE_BLOB must be a nonzero 40-hex Git blob'
valid_hex "${EXPECTED_COMPILER_SHA256:-}" 64 || unconfigured 'EXPECTED_COMPILER_SHA256 must be a nonzero 64-hex SHA256'
for program in git sha256sum cut cmp mktemp timeout tr; do
  command -v "$program" >/dev/null 2>&1 || unconfigured "missing required host command: $program"
done

root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
[ -f "$root/src/linear.ores" ] || unconfigured 'src/linear.ores missing'
[ -f "$root/tests/smoke.ores" ] || unconfigured 'tests/smoke.ores missing'
head="$(git -C "$root" rev-parse --verify HEAD 2>/dev/null)" || unconfigured 'source is not at an inspectable Git commit'
[ "$head" = "$EXPECTED_ML_HEAD_SHA" ] || unconfigured 'source HEAD mismatch'

linear="$(git hash-object "$root/src/linear.ores")"
smoke="$(git hash-object "$root/tests/smoke.ores")"
[ "$linear" = "$EXPECTED_LINEAR_BLOB" ] || unconfigured 'src/linear.ores blob mismatch'
[ "$smoke" = "$EXPECTED_SMOKE_BLOB" ] || unconfigured 'tests/smoke.ores blob mismatch'
compiler="$(sha256sum "$ORESLANG_BIN" | cut -d ' ' -f 1)"
[ "$compiler" = "$EXPECTED_COMPILER_SHA256" ] || unconfigured 'compiler binary SHA256 mismatch'

tmp="$(mktemp -d "${TMPDIR:-/tmp}/ores-ml-smoke.XXXXXXXX")" || unconfigured 'cannot create private test directory'
trap 'rm -rf -- "$tmp"' EXIT HUP INT TERM
printf 'ORES_ML_TRAINING_SMOKE_PASS\n' > "$tmp/expected"
if (cd "$root" && timeout 120s "$ORESLANG_BIN" tests/smoke.ores) > "$tmp/stdout" 2> "$tmp/stderr"; then
  :
else
  status=$?
  printf 'GUEST_FAILED: compiler exited %s (not a native pass)\n' "$status" >&2
  exit 1
fi
if [ -s "$tmp/stderr" ] || ! cmp -s "$tmp/expected" "$tmp/stdout"; then
  echo 'GUEST_FAILED: stdout was not the exact smoke sentinel or stderr was nonempty' >&2
  exit 1
fi
# Check source and fixture *again*: a guest process can overwrite the fixture
# after the preflight hash and still emit a matching success sentinel.
linear_after="$(git hash-object "$root/src/linear.ores")" || unconfigured 'cannot rehash affine source'
smoke_after="$(git hash-object "$root/tests/smoke.ores")" || unconfigured 'cannot rehash smoke fixture'
[ "$linear_after" = "$EXPECTED_LINEAR_BLOB" ] || unconfigured 'affine source mutated during execution'
[ "$smoke_after" = "$EXPECTED_SMOKE_BLOB" ] || unconfigured 'smoke fixture mutated during execution'
compiler_after="$(sha256sum "$ORESLANG_BIN" | cut -d ' ' -f 1)"
[ "$compiler_after" = "$EXPECTED_COMPILER_SHA256" ] || unconfigured 'compiler binary changed during test'
printf 'UNAUTHENTICATED_GUEST_OUTPUT_MATCH source=%s fixture=%s compiler_sha256=%s\n' "$head" "$smoke" "$compiler"
