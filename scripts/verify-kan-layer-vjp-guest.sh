#!/usr/bin/env bash
# KANLayer CPU packed backward VJP: actual guest gate, no JS fallback.
# A missing compiler or an unapproved CI carrier is NOT_EXECUTED, never PASS.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source="examples/kan_layer_vjp_f64_smoke.ores"
kernel="src/kan/clamped_cubic_bspline_f64.ores"

if [[ -z "${ORESLANG_BIN:-}" || ! -f "${ORESLANG_BIN:-}" || ! -x "${ORESLANG_BIN:-}" || "${ORESLANG_BIN}" != /* ]]; then
  echo "NOT_EXECUTED: ORESLANG_BIN must be an absolute executable file" >&2
  exit 2
fi
if [[ ! "${ORESLANG_BIN_SHA256:-}" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "NOT_EXECUTED: require verified 64-digit ORESLANG_BIN_SHA256" >&2
  exit 2
fi
if [[ ! "${EXPECTED_SOURCE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "NOT_EXECUTED: require an independently pinned source commit SHA" >&2
  exit 2
fi
if [[ "$(git -C "$root" rev-parse HEAD)" != "$EXPECTED_SOURCE_COMMIT" ]]; then
  echo "NOT_EXECUTED: source HEAD does not match expected commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$root" status --porcelain --untracked-files=all)" ]]; then
  echo "NOT_EXECUTED: source worktree is not clean" >&2
  exit 2
fi
if [[ "$(git -C "$root" hash-object -- "$kernel")" != "f49cc2ef36f808a18c5304c2a536497bb8d4759b" ]]; then
  echo "NOT_EXECUTED: multi-span Oreslang kernel Git blob changed" >&2
  exit 2
fi
if [[ "$(git -C "$root" hash-object -- "$source")" != "d2089c9653deddc1652816dff46d549fd3460adc" ]]; then
  echo "NOT_EXECUTED: multi-span Oreslang fixture Git blob changed" >&2
  exit 2
fi
if command -v sha256sum >/dev/null 2>&1; then
  compiler_digest="$(sha256sum -- "$ORESLANG_BIN" | cut -d ' ' -f1)"
elif command -v shasum >/dev/null 2>&1; then
  compiler_digest="$(shasum -a 256 -- "$ORESLANG_BIN" | cut -d ' ' -f1)"
else
  echo "NOT_EXECUTED: SHA256 tool unavailable" >&2
  exit 2
fi
if [[ "${compiler_digest,,}" != "${ORESLANG_BIN_SHA256,,}" ]]; then
  echo "NOT_EXECUTED: compiler executable digest mismatch" >&2
  exit 2
fi
if command -v timeout >/dev/null 2>&1; then
  timeout_bin="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
  timeout_bin="$(command -v gtimeout)"
else
  echo "NOT_EXECUTED: GNU timeout is required" >&2
  exit 2
fi
duration="${ORESLANG_GUEST_TIMEOUT_SECONDS:-120}"
if [[ ! "$duration" =~ ^[1-9][0-9]{0,2}$ ]] || (( 10#$duration > 600 )); then
  echo "NOT_EXECUTED: invalid guest timeout (1..600 seconds)" >&2
  exit 2
fi

tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
printf '%s\n' "ORES_KAN_LAYER_VJP_PASS" > "$tmp/expected"
if ! "$timeout_bin" -k 5s "${duration}s" "$ORESLANG_BIN" "$root/$source" > "$tmp/stdout" 2> "$tmp/stderr"; then
  echo "GUEST=FAILED: compiler exited nonzero or timed out" >&2
  exit 1
fi
if [[ -s "$tmp/stderr" ]]; then
  echo "GUEST=FAILED: unexpected stderr" >&2
  exit 1
fi
if ! cmp -s "$tmp/expected" "$tmp/stdout"; then
  echo "GUEST=OUTPUT_MISMATCH: no pass receipt" >&2
  exit 1
fi
printf 'GUEST=PASS; SOURCE_COMMIT=%s; COMPILER_SHA256=%s; GUEST_FIXTURE_BLOB=d2089c9653deddc1652816dff46d549fd3460adc; KAN_KERNEL_BLOB=f49cc2ef36f808a18c5304c2a536497bb8d4759b\n' "$EXPECTED_SOURCE_COMMIT" "$compiler_digest"
