#!/usr/bin/env bash
# MOCK-only orchestration tests. Never compile guests or export private source.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sandbox="$(mktemp -d)"
trap 'rm -rf -- "$sandbox"' EXIT
mkdir -p "$sandbox/scripts"
cp "$root/scripts/verify-kan-native-cpu-suite.sh" "$sandbox/scripts/"
for name in verify-kan-multispan-guest.sh verify-kan-layer-guest.sh verify-kan-layer-vjp-guest.sh; do
  cat > "$sandbox/scripts/$name" <<'CHILD'
#!/usr/bin/env bash
case "${MOCK_MODE:-pass}" in
  fail) if [[ "$(basename "$0")" == 'verify-kan-layer-guest.sh' ]]; then exit 2; fi ;;
  bad) if [[ "$(basename "$0")" == 'verify-kan-layer-guest.sh' ]]; then
    printf '%s\n' 'GUEST=PASS; SOURCE_COMMIT=WRONG; COMPILER_SHA256=WRONG;'
    exit 0
  fi ;;
  multi) if [[ "$(basename "$0")" == 'verify-kan-layer-guest.sh' ]]; then
    printf '%s\n' 'GUEST=PASS; SOURCE_COMMIT=WRONG; COMPILER_SHA256=WRONG;' 'EXTRA_OUTPUT'
    exit 0
  fi ;;
esac
printf 'GUEST=PASS; SOURCE_COMMIT=%s; COMPILER_SHA256=%s; KERNEL_BLOB=MOCK_ONLY\n' \
  "$EXPECTED_SOURCE_COMMIT" "${ORESLANG_BIN_SHA256,,}"
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
    printf 'FAIL receipt harness expected rc=%s substring=%s got rc=%s output=%s\n' \
      "$expected_status" "$expected_text" "$rc" "$output" >&2
    exit 1
  fi
  (( count += 1 ))
}
check 2 'source commit SHA required'
check 2 'compiler SHA256 required' EXPECTED_SOURCE_COMMIT="$sha"
check 2 'real Oreslang compiler required' EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest"
check 0 'KAN_CPU_SUITE=PASS; GUEST_FIXTURES=3' EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN=/mock
check 2 'KAN_CPU_SUITE=NOT_PASS' MOCK_MODE=fail EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN=/mock
check 1 'INVALID_RECEIPT' MOCK_MODE=bad EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN=/mock
check 1 'INVALID_RECEIPT' MOCK_MODE=multi EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN=/mock
printf 'MOCK_NATIVE_RECEIPT_SUITE=PASS; TESTS=%d; ORESLANG_GUEST=NOT_EXECUTED\n' "$count"
