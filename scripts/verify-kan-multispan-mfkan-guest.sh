#!/usr/bin/env bash
# Real Oreslang CPU guest gate. No host mathematical substitute is accepted.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source="examples/kan_multispan_mfkan_smoke.ores"
mfkan="src/kan/mfkan_clamped_cubic_f64.ores"
spline="src/kan/clamped_cubic_bspline_f64.ores"
if [[ -z "${ORESLANG_BIN:-}" || "${ORESLANG_BIN}" != /* || ! -f "${ORESLANG_BIN}" || ! -x "${ORESLANG_BIN}" ]]; then
  echo "NOT_EXECUTED: ORESLANG_BIN must be an absolute executable file" >&2
  exit 2
fi
if [[ ! "${ORESLANG_BIN_SHA256:-}" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "NOT_EXECUTED: require independently pinned ORESLANG_BIN_SHA256" >&2
  exit 2
fi
if [[ ! "${EXPECTED_SOURCE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "NOT_EXECUTED: require independently pinned EXPECTED_SOURCE_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$root" rev-parse HEAD)" != "$EXPECTED_SOURCE_COMMIT" ]]; then
  echo "NOT_EXECUTED: source HEAD differs from pinned commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$root" status --porcelain --untracked-files=all)" ]]; then
  echo "NOT_EXECUTED: dirty source checkout" >&2
  exit 2
fi
if [[ "$(git -C "$root" hash-object -- "$spline")" != "78f1b36287154ca81107d67fa94f5e7275b3caa2" ||
      "$(git -C "$root" hash-object -- "$mfkan")" != "08c145c8e2224e330907100ef6897784a23c1af5" ||
      "$(git -C "$root" hash-object -- "$source")" != "7e979dedd63ea861387c9c64b4092da9f1e8af4d" ]]; then
  echo "NOT_EXECUTED: a reviewed Oreslang source blob differs" >&2
  exit 2
fi
if command -v sha256sum >/dev/null 2>&1; then
  digest="$(sha256sum -- "$ORESLANG_BIN" | cut -d ' ' -f1)"
elif command -v shasum >/dev/null 2>&1; then
  digest="$(shasum -a 256 -- "$ORESLANG_BIN" | cut -d ' ' -f1)"
else
  echo "NOT_EXECUTED: SHA256 tool unavailable" >&2
  exit 2
fi
if [[ "${digest,,}" != "${ORESLANG_BIN_SHA256,,}" ]]; then
  echo "NOT_EXECUTED: compiler SHA256 mismatch" >&2
  exit 2
fi
if command -v timeout >/dev/null 2>&1; then
  timeout_bin="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
  timeout_bin="$(command -v gtimeout)"
else
  echo "NOT_EXECUTED: GNU timeout required" >&2
  exit 2
fi
duration="${ORESLANG_GUEST_TIMEOUT_SECONDS:-120}"
if [[ ! "$duration" =~ ^[1-9][0-9]{0,2}$ ]] || (( 10#$duration > 600 )); then
  echo "NOT_EXECUTED: guest timeout must be 1..600 seconds" >&2
  exit 2
fi
temp="$(mktemp -d)"
trap 'rm -rf -- "$temp"' EXIT
printf '%s\n' "ORES_MFKAN_MULTISPAN_PASS" > "$temp/expected"
if ! "$timeout_bin" -k 5s "${duration}s" "$ORESLANG_BIN" "$root/$source" >"$temp/stdout" 2>"$temp/stderr"; then
  echo "GUEST=FAILED: compiler exit failure or timeout" >&2
  exit 1
fi
if [[ -s "$temp/stderr" ]]; then
  echo "GUEST=FAILED: unexpected stderr" >&2
  exit 1
fi
if ! cmp -s "$temp/expected" "$temp/stdout"; then
  echo "GUEST=OUTPUT_MISMATCH: no exact acceptance receipt" >&2
  exit 1
fi
printf 'GUEST=PASS; SOURCE_COMMIT=%s; COMPILER_SHA256=%s; MFKAN_BLOB=08c145c8e2224e330907100ef6897784a23c1af5; SMOKE_BLOB=7e979dedd63ea861387c9c64b4092da9f1e8af4d\n' "$EXPECTED_SOURCE_COMMIT" "$digest"
