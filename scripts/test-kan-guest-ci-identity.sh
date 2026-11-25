#!/usr/bin/env bash
# Source-free pre-compiler, fail-closed CI identity admission tests.
# This neither executes an Oreslang guest nor authorizes cross-org transfer.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pass=0
check() {
  local expected="$1"
  local script="$2"
  shift 2
  local output status=0
  output="$(env -i PATH="$PATH" HOME="${HOME:-/tmp}" "$@" bash "$root/scripts/$script" 2>&1)" || status=$?
  if [[ "$status" -ne 2 || "$output" != *"$expected"* ]]; then
    printf 'FAIL %s expected %s (status %s) got %s\n' "$script" "$expected" "$status" "$output" >&2
    exit 1
  fi
  (( pass += 1 ))
}
for script in \
  verify-kan-multispan-guest.sh \
  verify-kan-layer-guest.sh \
  verify-kan-layer-vjp-guest.sh \
  verify-kan-layer-vjp-extended-guest.sh; do
  check 'CI source repository identity mismatch' "$script" \
    GITHUB_ACTIONS=true GITHUB_REPOSITORY=other-org/private-ci GITHUB_REPOSITORY_ID=1448720923
  check 'CI source repository identity mismatch' "$script" \
    GITHUB_ACTIONS=true GITHUB_REPOSITORY=atto-lang/atto-ml GITHUB_REPOSITORY_ID=99999
  check 'CI source repository identity mismatch' "$script" \
    GITHUB_ACTIONS=true GITHUB_REPOSITORY=atto-lang/atto-ml
  check 'unverifiable CI source repository context' "$script" \
    CI=true
  check 'unverifiable CI source repository context' "$script" \
    GITHUB_REPOSITORY=atto-lang/atto-ml
  check 'unverifiable CI source repository context' "$script" \
    CI=1
  check 'unverifiable CI source repository context' "$script" \
    GITHUB_ACTIONS=false
  # Valid identity moves to the independent mandatory compiler preflight.
  check 'NOT_EXECUTED:' "$script" \
    GITHUB_ACTIONS=true GITHUB_REPOSITORY=atto-lang/atto-ml GITHUB_REPOSITORY_ID=1448720923
  check 'NOT_EXECUTED:' "$script"
done
printf 'CI_GUARD_HOST_ONLY=PASS; TESTS=%d; ORESLANG_GUEST=NOT_EXECUTED\n' "$pass"
