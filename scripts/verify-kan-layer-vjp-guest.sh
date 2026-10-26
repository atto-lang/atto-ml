#!/usr/bin/env bash
# Actual native guest, not a JavaScript reference fallback.
# Private source is allowed only in source org unless a CI carrier is approved.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source="examples/kan_layer_vjp_smoke.ores"
kernel="src/kan/packed_layer_vjp_f64.ores"
# Source-CI identity sanity check, not authorization to transfer private source.
if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
  if [[ "${GITHUB_REPOSITORY:-}" != "ores-truffle-oreslang/oreslang-ml" \
        || "${GITHUB_REPOSITORY_ID:-}" != "1423913534" ]]; then
    echo 'NOT_EXECUTED: CI source repository identity mismatch' >&2
    exit 2
  fi
elif [[ -n "${GITHUB_ACTIONS:-}" || -n "${GITHUB_REPOSITORY:-}" \
        || -n "${GITHUB_REPOSITORY_ID:-}" || -n "${CI:-}" ]]; then
  echo 'NOT_EXECUTED: unverifiable CI source repository context' >&2
  exit 2
fi
if [[ -z "${ORESLANG_BIN:-}" || "${ORESLANG_BIN}" != /* || ! -x "${ORESLANG_BIN}" || ! -f "${ORESLANG_BIN}" ]]; then
  echo 'NOT_EXECUTED: no absolute executable compiler' >&2; exit 2
fi
if [[ ! "${ORESLANG_BIN_SHA256:-}" =~ ^[a-fA-F0-9]{64}$ || ! "${EXPECTED_SOURCE_COMMIT:-}" =~ ^[a-f0-9]{40}$ ]]; then
  echo 'NOT_EXECUTED: require trusted compiler sha and exact source commit' >&2; exit 2
fi
command -v git >/dev/null || { echo 'NOT_EXECUTED: git missing' >&2; exit 2; }
if [[ "$(git -C "$root" rev-parse HEAD)" != "$EXPECTED_SOURCE_COMMIT" ]]; then
  echo 'NOT_EXECUTED: mismatched source HEAD' >&2; exit 2
fi
if [[ -n "$(git -C "$root" status --porcelain --untracked-files=all)" ]]; then
  echo 'NOT_EXECUTED: dirty checkout' >&2; exit 2
fi
if [[ "$(git -C "$root" hash-object -- "$kernel")" != 'b3a36931e8fc1c3d5f4912b42bd4fcdcc2aab704' ]]; then
  echo 'NOT_EXECUTED: mismatched kernel blob' >&2; exit 2
fi
if [[ "$(git -C "$root" hash-object -- "$source")" != '2e19be4571f0c77eb782867972708ae3a35f4656' ]]; then
  echo 'NOT_EXECUTED: mismatched guest fixture blob' >&2; exit 2
fi
if command -v sha256sum >/dev/null 2>&1; then
  compiler_hash="$(sha256sum -- "$ORESLANG_BIN" | cut -d ' ' -f1)"
elif command -v shasum >/dev/null 2>&1; then
  compiler_hash="$(shasum -a 256 -- "$ORESLANG_BIN" | cut -d ' ' -f1)"
else
  echo 'NOT_EXECUTED: SHA256 tool missing' >&2; exit 2
fi
if [[ "${compiler_hash,,}" != "${ORESLANG_BIN_SHA256,,}" ]]; then
  echo 'NOT_EXECUTED: compiler SHA256 mismatch' >&2; exit 2
fi
if command -v timeout >/dev/null 2>&1; then
  timeout_bin="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
  timeout_bin="$(command -v gtimeout)"
else
  echo 'NOT_EXECUTED: GNU timeout unavailable' >&2; exit 2
fi
seconds="${ORESLANG_GUEST_TIMEOUT_SECONDS:-120}"
if [[ ! "$seconds" =~ ^[1-9][0-9]{0,2}$ ]] || (( 10#$seconds > 600 )); then
  echo 'NOT_EXECUTED: timeout outside 1..600s' >&2; exit 2
fi
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
printf '%s\n' 'ORES_KAN_LAYER_VJP_PASS' > "$tmp/expected"
if ! "$timeout_bin" -k 5s "${seconds}s" "$ORESLANG_BIN" "$root/$source" > "$tmp/actual" 2> "$tmp/stderr"; then
  echo 'GUEST=FAILED: compilation/execution failure or timeout' >&2; exit 1
fi
if [[ -s "$tmp/stderr" ]] || ! cmp -s "$tmp/expected" "$tmp/actual"; then
  echo 'GUEST=FAILED: unexpected stderr or stdout' >&2; exit 1
fi
printf 'GUEST=PASS; SOURCE_COMMIT=%s; COMPILER_SHA256=%s; KERNEL_BLOB=b3a36931e8fc1c3d5f4912b42bd4fcdcc2aab704; FIXTURE_BLOB=2e19be4571f0c77eb782867972708ae3a35f4656\n' "$EXPECTED_SOURCE_COMMIT" "$compiler_hash"
