#!/usr/bin/env bash
# MOCK-ONLY acceptance of the aggregate verifier; uses an isolated test Git
# repository and a hashed dummy executable. NEVER compiles or runs Oreslang.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sandbox="$(mktemp -d)"
trap 'rm -rf -- "$sandbox"' EXIT
work="$sandbox/repo"
mkdir -p "$work/scripts"
cp "$root/scripts/verify-kan-native-cpu-suite.sh" "$work/scripts/"
for name in verify-kan-multispan-guest.sh verify-kan-layer-guest.sh verify-kan-layer-vjp-guest.sh verify-kan-layer-vjp-extended-guest.sh; do
  cat > "$work/scripts/$name" <<'CHILD'
#!/usr/bin/env bash
set -euo pipefail
case "$(basename "$0")" in
  verify-kan-multispan-guest.sh)
    suffix='GUEST_FIXTURE_BLOB=93732a26c708c1e06604c6b5d1127ed87fe3ba28; KAN_KERNEL_BLOB=f49cc2ef36f808a18c5304c2a536497bb8d4759b'
    ;;
  verify-kan-layer-guest.sh)
    suffix='GUEST_FIXTURE_BLOB=3db364a880500c584083eb1c7eea7defea1c6a26; KAN_KERNEL_BLOB=f49cc2ef36f808a18c5304c2a536497bb8d4759b'
    ;;
  verify-kan-layer-vjp-guest.sh)
    suffix='GUEST_FIXTURE_BLOB=d2089c9653deddc1652816dff46d549fd3460adc; KAN_KERNEL_BLOB=f49cc2ef36f808a18c5304c2a536497bb8d4759b'
    ;;
  verify-kan-layer-vjp-extended-guest.sh)
    suffix='GUEST_FIXTURE_BLOB=baf209365c98c83ef8204868749f08e4bc69a0e2; KAN_KERNEL_BLOB=f49cc2ef36f808a18c5304c2a536497bb8d4759b'
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
# Use real Git and actual SHA256 operations to exercise the production preflight.
# The dummy compiler is never invoked: only its bytes are hashed.
git -C "$work" init -q
git -C "$work" -c user.name='mock' -c user.email='mock@example.invalid' add scripts
git -C "$work" -c user.name='mock' -c user.email='mock@example.invalid' commit -qm 'isolated source-free mock fixture'
sha="$(git -C "$work" rev-parse HEAD)"
printf '%s\n' '#!/usr/bin/env bash' 'exit 11' > "$sandbox/dummy-compiler"
chmod 700 "$sandbox/dummy-compiler"
compiler="$sandbox/dummy-compiler"
if command -v sha256sum >/dev/null 2>&1; then
  digest="$(sha256sum "$compiler" | cut -d ' ' -f1)"
else
  digest="$(shasum -a 256 "$compiler" | cut -d ' ' -f1)"
fi
count=0
check() {
  local expected_status="$1" expected_text="$2"
  shift 2
  local output rc=0
  output="$(env -i PATH="$PATH" HOME="${HOME:-/tmp}" "$@" bash "$work/scripts/verify-kan-native-cpu-suite.sh" 2>&1)" || rc=$?
  if [[ "$rc" != "$expected_status" || "$output" != *"$expected_text"* ]]; then
    printf 'FAIL expected rc=%s text=%s, got rc=%s output=%s\n' \
      "$expected_status" "$expected_text" "$rc" "$output" >&2
    exit 1
  fi
  ((count += 1))
}
check 2 'source commit SHA required'
check 2 'compiler SHA256 required' EXPECTED_SOURCE_COMMIT="$sha"
check 2 'absolute executable' EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest"
check 2 'absolute executable' EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN=/mock
check 2 'source HEAD differs' EXPECTED_SOURCE_COMMIT=0000000000000000000000000000000000000000 ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN="$compiler"
check 2 'compiler SHA256 differs' EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff ORESLANG_BIN="$compiler"
check 2 'CI source repository identity mismatch' GITHUB_ACTIONS=true GITHUB_REPOSITORY=other/private GITHUB_REPOSITORY_ID=1448720923 EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN="$compiler"
check 2 'unverifiable CI source repository context' CI=1 EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN="$compiler"
check 0 'KAN_CPU_SUITE=PASS; GUEST_FIXTURES=4' EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN="$compiler"
check 0 'KAN_CPU_SUITE=PASS; GUEST_FIXTURES=4' GITHUB_ACTIONS=true GITHUB_REPOSITORY=atto-lang/atto-ml GITHUB_REPOSITORY_ID=1448720923 EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN="$compiler"
check 2 'KAN_CPU_SUITE=NOT_PASS' MOCK_MODE=fail EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN="$compiler"
for mode in bad multi blank wrong_kernel wrong_fixture no_hashes trailing stderr; do
  check 1 'INVALID_RECEIPT' MOCK_MODE="$mode" EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN="$compiler"
done
printf 'untracked\n' > "$work/dirty-file"
check 2 'dirty source checkout' EXPECTED_SOURCE_COMMIT="$sha" ORESLANG_BIN_SHA256="$digest" ORESLANG_BIN="$compiler"
rm "$work/dirty-file"
printf 'MOCK_NATIVE_RECEIPT_SUITE=PASS; TESTS=%d; ORESLANG_GUEST=NOT_EXECUTED\n' "$count"
