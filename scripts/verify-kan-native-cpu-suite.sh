#!/usr/bin/env bash
# Native Oreslang CPU fixtures only. Not a host arithmetic fallback.
# Requires a source-authorized runner and independently pinned compiler/toolchain.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! "${EXPECTED_SOURCE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'NOT_EXECUTED: exact trusted source commit SHA required' >&2; exit 2
fi
if [[ ! "${ORESLANG_BIN_SHA256:-}" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo 'NOT_EXECUTED: independently pinned compiler SHA256 required' >&2; exit 2
fi
if [[ -z "${ORESLANG_BIN:-}" ]]; then
  echo 'NOT_EXECUTED: real Oreslang compiler required' >&2; exit 2
fi
expected="GUEST=PASS; SOURCE_COMMIT=${EXPECTED_SOURCE_COMMIT}; COMPILER_SHA256=${ORESLANG_BIN_SHA256,,};"
count=0
for script in \
  verify-kan-multispan-guest.sh \
  verify-kan-layer-guest.sh \
  verify-kan-layer-vjp-guest.sh; do
  receipt=''
  if receipt="$(bash "$root/scripts/$script")"; then
    :
  else
    status=$?
    printf 'KAN_CPU_SUITE=NOT_PASS; COMPONENT=%s; EXIT_CODE=%s\n' "$script" "$status" >&2
    exit "$status"
  fi
  if [[ "$receipt" != "$expected"* || "$receipt" == *$'\n'* ]]; then
    printf 'KAN_CPU_SUITE=NOT_PASS; COMPONENT=%s; INVALID_RECEIPT\n' "$script" >&2
    exit 1
  fi
  printf 'COMPONENT=%s; %s\n' "$script" "$receipt"
  ((count += 1))
done
if [[ "$count" != 3 ]]; then
  echo 'KAN_CPU_SUITE=NOT_PASS; missing required native fixture' >&2; exit 1
fi
printf 'KAN_CPU_SUITE=PASS; GUEST_FIXTURES=%d; SOURCE_COMMIT=%s; COMPILER_SHA256=%s\n' \
  "$count" "$EXPECTED_SOURCE_COMMIT" "${ORESLANG_BIN_SHA256,,}"
