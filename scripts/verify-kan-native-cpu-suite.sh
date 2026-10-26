#!/usr/bin/env bash
# Native Oreslang guest integration: no host-language numerical fallback.
# Requires approved exact source HEAD and independently pinned compiler identity.
# Never treat the provenance fields printed by a child as compiler attestation.
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
prefix="GUEST=PASS; SOURCE_COMMIT=${EXPECTED_SOURCE_COMMIT}; COMPILER_SHA256=${ORESLANG_BIN_SHA256,,};"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
count=0
for script in \
  verify-kan-multispan-guest.sh \
  verify-kan-layer-guest.sh \
  verify-kan-layer-vjp-guest.sh; do
  case "$script" in
    verify-kan-multispan-guest.sh)
      receipt="$prefix GUEST_FIXTURE_BLOB=93732a26c708c1e06604c6b5d1127ed87fe3ba28; KAN_KERNEL_BLOB=322e344daf6c3474610cb04da05501cee99af70a"
      ;;
    verify-kan-layer-guest.sh)
      receipt="$prefix GUEST_FIXTURE_BLOB=3db364a880500c584083eb1c7eea7defea1c6a26; KAN_KERNEL_BLOB=322e344daf6c3474610cb04da05501cee99af70a"
      ;;
    verify-kan-layer-vjp-guest.sh)
      receipt="$prefix KERNEL_BLOB=b3a36931e8fc1c3d5f4912b42bd4fcdcc2aab704; FIXTURE_BLOB=2e19be4571f0c77eb782867972708ae3a35f4656"
      ;;
    *) echo 'KAN_CPU_SUITE=NOT_PASS; unexpected verifier' >&2; exit 2 ;;
  esac
  if bash "$root/scripts/$script" > "$tmp/actual" 2> "$tmp/stderr"; then
    :
  else
    status=$?
    printf 'KAN_CPU_SUITE=NOT_PASS; COMPONENT=%s; EXIT_CODE=%s\n' "$script" "$status" >&2
    cat "$tmp/stderr" >&2
    exit "$status"
  fi
  printf '%s\n' "$receipt" > "$tmp/expected"
  if [[ -s "$tmp/stderr" ]] || ! cmp -s "$tmp/expected" "$tmp/actual"; then
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
