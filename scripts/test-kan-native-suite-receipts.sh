#!/usr/bin/env bash
# MOCK-only fixture receipt validation; never executes a compiler or native guest.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sandbox="$(mktemp -d)"
trap 'rm -rf -- "$sandbox"' EXIT
mkdir -p "$sandbox/scripts"
cp "$root/scripts/verify-kan-native-cpu-suite.sh" "$sandbox/scripts/"
for name in verify-kan-multispan-guest.sh verify-kan-layer-guest.sh verify-kan-layer-vjp-guest.sh; do
  cat > "$sandbox/scripts/$name" <<'CHILD'
#!/usr/bin/env bash
set -euo pipefail
case "$(basename "$0")" in
  verify-kan-multispan-guest.sh)
    suffix='GUEST_FIXTURE_BLOB=93732a26c708c1e06604c6b5d1127ed87fe3ba28; KAN_KERNEL_BLOB=322e344daf6c3474610cb04da05501cee99af70a'
    ;;
  verify-kan-layer-guest.sh)
    suffix='GUEST_FIXTURE_BLOB=3db364a880500c584083eb1c7eea7defea1c6a26; KAN_KERNEL_BLOB=322e344daf6c3474610cb04da05501cee99af70a'
    ;;
  verify-kan-layer-vjp-guest.sh)
    suffix='KERNEL_BLOB=b3a36931e8fc1c3d5f4912b42bd4fcdcc2aab704; FIXTURE_BLOB=2e19be4571f0c77eb782867972708ae3a35f4656'
    ;;
esac
receipt="GUEST=PASS; SOURCE_COMMIT=${EXPECTED_SOURCE_COMMIT}; COMPILER_SHA256=${ORESLANG_BIN_SHA256,,}; $suffix"
if [[ "$(basename "$0")" == "verify-kan-layer-guest.sh" ]]; then
  case "${MOCK_MODE:-pass}" in
    fail) exit 2 ;;
    bad) receipt='GUEST=PASS; SOURCE_COMMIT=WRONG; COMPILER_SHA256=WRONG;' ;;
    multi) printf '%s\n' "$receipt" 'EXTRA_OUTPUT'; exit 0 ;;
    blank) printf '%s\n\n' "$receipt"; exit 0 ;;
    wrong_kernel) receipt="${receipt%KAN_KERNEL_BLOB=*}KAN_KERNEL_BLOB=0000000000000000000000000000000000000000" ;;
    wrong_fixture) receipt="${receipt/GUEST_FIXTURE_BLOB=3db364a880500c584083eb1c7eea7defea1c6a26/GUEST_FIXTURE_BLOB=1111111111111111111111111111111111111111}" ;;
    no_hashes) receipt="GUEST=PASS; SOURCE_COMMIT=${EXPECTED_SOURCE_COMMIT}; COMPILER_SHA256=${ORESLANG_BIN_SHA256,,};" ;;
    trailing) receipt="$receipt; unapproved extra metadata" ;;
    stderr) printf '%s\n' 'UNEXPECTED_DIAGNOSTIC' >&2 ;;
  esac
fi
printf '%s\n' "$receipt"
CHILD
done
sha=0123456789abcdef0123456789abcdef01234567
digest=abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789
count=0
check() {
  local expected_status="$1" expected_text="$2"
  shift 2
  local output rc=0
  output="$(env -i PATH="$PATH" HOME="${HOME:-/tmp}" "$@" bash "$sandbox/scripts/verify-kan-native-cpu-suite.sh" 2>&1)" || rc=$?
  if [[ "$rc" != "$expected_status" || "$output" != *"$expected_text"* ]]; then
    printf 'FAIL expected rc=%s text=%s got rc=%s output=%s\n' \
      "$expected_status" "$expected_text" "$rc" "$output" >&2
    exit 1
  fi
  ((count += 1))
}
check 2 'source commit SHA required'
check 2 'compiler SHA256 required' EXPECTED_SOURCE_COMMIT="$sha"
check 2 'real Oreslang compiler required' EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest"
check 0 'KAN_CPU_SUITE=PASS; GUEST_FIXTURES=3' EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN=/mock
check 2 'KAN_CPU_SUITE=NOT_PASS' MOCK_MODE=fail EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN=/mock
for mode in bad multi blank wrong_kernel wrong_fixture no_hashes trailing stderr; do
  check 1 'INVALID_RECEIPT' MOCK_MODE="$mode" EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN=/mock
done
printf 'MOCK_NATIVE_RECEIPT_SUITE=PASS; TESTS=%d; ORESLANG_GUEST=NOT_EXECUTED\n' "$count"
