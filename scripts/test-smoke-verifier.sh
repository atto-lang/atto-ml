#!/usr/bin/env sh
# Source-free verifier self-test. All fake binaries are intentionally NOT Oreslang.
set -eu
root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/ores-ml-verifier-selftest.XXXXXXXX")"
trap 'rm -rf -- "$tmp"' EXIT HUP INT TERM
mkdir -p "$tmp/work/scripts" "$tmp/work/src" "$tmp/work/tests"
cp "$root/scripts/test.sh" "$tmp/work/scripts/test.sh"
printf 'dummy affine kernel\n' > "$tmp/work/src/linear.ores"
printf 'dummy smoke fixture\n' > "$tmp/work/tests/smoke.ores"
git -C "$tmp/work" init -q
git -C "$tmp/work" add .
git -C "$tmp/work" -c user.name=VerifierSelftest -c user.email=unused@example.invalid commit -qm fixture
head="$(git -C "$tmp/work" rev-parse HEAD)"
linear="$(git hash-object "$tmp/work/src/linear.ores")"
smoke="$(git hash-object "$tmp/work/tests/smoke.ores")"
cat > "$tmp/mock-accept" <<'MOCK'
#!/bin/sh
printf 'ORES_ML_TRAINING_SMOKE_PASS\n'
MOCK
cat > "$tmp/mock-noise" <<'MOCK'
#!/bin/sh
printf 'ORES_ML_TRAINING_SMOKE_PASS\n'
printf 'unexpected stderr\n' >&2
MOCK
cat > "$tmp/mock-extra" <<'MOCK'
#!/bin/sh
printf 'ORES_ML_TRAINING_SMOKE_PASS\nextra\n'
MOCK
cat > "$tmp/mock-nonzero" <<'MOCK'
#!/bin/sh
printf 'ORES_ML_TRAINING_SMOKE_PASS\n'
exit 7
MOCK
chmod 700 "$tmp"/mock-*
script="$tmp/work/scripts/test.sh"
sha_true="$(sha256sum /bin/true | cut -d ' ' -f 1)"
sha_good="$(sha256sum "$tmp/mock-accept" | cut -d ' ' -f 1)"
sha_noise="$(sha256sum "$tmp/mock-noise" | cut -d ' ' -f 1)"
sha_extra="$(sha256sum "$tmp/mock-extra" | cut -d ' ' -f 1)"
sha_nonzero="$(sha256sum "$tmp/mock-nonzero" | cut -d ' ' -f 1)"
assert_rc() {
  expected="$1"
  shift
  set +e
  "$@" > "$tmp/actual-stdout" 2> "$tmp/actual-stderr"
  actual="$?"
  set -e
  if [ "$actual" -ne "$expected" ]; then
    printf 'FAIL: expected exit=%s got=%s\n' "$expected" "$actual" >&2
    cat "$tmp/actual-stderr" >&2
    exit 1
  fi
}
run_case() {
  expected="$1"
  binary="$2"
  hash="$3"
  head_pin="$4"
  linear_pin="$5"
  smoke_pin="$6"
  assert_rc "$expected" env "ORESLANG_BIN=$binary" "EXPECTED_COMPILER_SHA256=$hash" \
    "EXPECTED_ML_HEAD_SHA=$head_pin" "EXPECTED_LINEAR_BLOB=$linear_pin" \
    "EXPECTED_SMOKE_BLOB=$smoke_pin" sh "$script"
}
# No pins, no absolute executable, forged source or fixture, and wrong binary
# must all fail closed with UNCONFIGURED exit 78.
assert_rc 78 env ORESLANG_BIN='' sh "$script"
run_case 78 true "$sha_good" "$head" "$linear" "$smoke"
run_case 78 "$tmp/mock-accept" "$sha_good" "$(printf '%040d' 1)" "$linear" "$smoke"
run_case 78 "$tmp/mock-accept" "$sha_good" "$head" "$(printf '%040d' 1)" "$smoke"
run_case 78 "$tmp/mock-accept" "$sha_good" "$head" "$linear" "$(printf '%040d' 1)"
run_case 78 "$tmp/mock-accept" "$sha_true" "$head" "$linear" "$smoke"
# An executable that exits zero but does not emit the exact marker is NOT a pass.
run_case 1 /bin/true "$sha_true" "$head" "$linear" "$smoke"
run_case 1 "$tmp/mock-noise" "$sha_noise" "$head" "$linear" "$smoke"
run_case 1 "$tmp/mock-extra" "$sha_extra" "$head" "$linear" "$smoke"
run_case 1 "$tmp/mock-nonzero" "$sha_nonzero" "$head" "$linear" "$smoke"
# Positive structural case deliberately uses a FAKE compiler: output+hash
# matches are NOT compiler attestation or a native guest execution proof.
run_case 0 "$tmp/mock-accept" "$sha_good" "$head" "$linear" "$smoke"
grep -q '^UNAUTHENTICATED_GUEST_OUTPUT_MATCH ' "$tmp/actual-stdout"
echo 'SOURCE_ONLY_SMOKE_VERIFIER_SELFTEST_PASS cases=11 native_guest=NOT_EXECUTED'
