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
printf 'dummy extra tracked file\n' > "$tmp/work/tests/other.txt"
git -C "$tmp/work" init -q
git -C "$tmp/work" add .
git -C "$tmp/work" -c user.name=VerifierSelftest -c user.email=unused@example.invalid commit -qm fixture
head="$(git -C "$tmp/work" rev-parse HEAD)"
tree="$(git -C "$tmp/work" rev-parse HEAD^{tree})"
linear="$(git hash-object "$tmp/work/src/linear.ores")"
smoke="$(git hash-object "$tmp/work/tests/smoke.ores")"
cat > "$tmp/mock-accept" <<'MOCK'
#!/bin/sh
printf 'ORES_ML_TRAINING_SMOKE_PASS\n'
MOCK
cat > "$tmp/mock-no-output" <<'MOCK'
#!/bin/sh
exit 0
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
cat > "$tmp/mock-source-mutate" <<'MOCK'
#!/bin/sh
printf 'tampered affine kernel\n' > src/linear.ores
printf 'ORES_ML_TRAINING_SMOKE_PASS\n'
MOCK
cat > "$tmp/mock-fixture-mutate" <<'MOCK'
#!/bin/sh
printf 'tampered smoke fixture\n' > tests/smoke.ores
printf 'ORES_ML_TRAINING_SMOKE_PASS\n'
MOCK
cat > "$tmp/mock-add-untracked" <<'MOCK'
#!/bin/sh
printf 'rogue' > src/rogue.ores
printf 'ORES_ML_TRAINING_SMOKE_PASS\n'
MOCK
cat > "$tmp/mock-extra-mutate" <<'MOCK'
#!/bin/sh
printf 'tampered extra file\n' > tests/other.txt
printf 'ORES_ML_TRAINING_SMOKE_PASS\n'
MOCK
cat > "$tmp/mock-empty-commit" <<'MOCK'
#!/bin/sh
git -c user.name=VerifierSelftest -c user.email=unused@example.invalid commit --allow-empty -qm changed-ref
printf 'ORES_ML_TRAINING_SMOKE_PASS\n'
MOCK
cat > "$tmp/mock-tree-commit" <<'MOCK'
#!/bin/sh
printf 'tampered extra file\n' > tests/other.txt
git add tests/other.txt
git -c user.name=VerifierSelftest -c user.email=unused@example.invalid commit -qm changed-tree
printf 'ORES_ML_TRAINING_SMOKE_PASS\n'
MOCK
chmod 700 "$tmp"/mock-*
script="$tmp/work/scripts/test.sh"
sha_no_output="$(sha256sum "$tmp/mock-no-output" | cut -d ' ' -f 1)"
sha_good="$(sha256sum "$tmp/mock-accept" | cut -d ' ' -f 1)"
sha_noise="$(sha256sum "$tmp/mock-noise" | cut -d ' ' -f 1)"
sha_extra="$(sha256sum "$tmp/mock-extra" | cut -d ' ' -f 1)"
sha_nonzero="$(sha256sum "$tmp/mock-nonzero" | cut -d ' ' -f 1)"
sha_source_mutate="$(sha256sum "$tmp/mock-source-mutate" | cut -d ' ' -f 1)"
sha_fixture_mutate="$(sha256sum "$tmp/mock-fixture-mutate" | cut -d ' ' -f 1)"
sha_add_untracked="$(sha256sum "$tmp/mock-add-untracked" | cut -d ' ' -f 1)"
sha_extra_mutate="$(sha256sum "$tmp/mock-extra-mutate" | cut -d ' ' -f 1)"
sha_empty_commit="$(sha256sum "$tmp/mock-empty-commit" | cut -d ' ' -f 1)"
sha_tree_commit="$(sha256sum "$tmp/mock-tree-commit" | cut -d ' ' -f 1)"
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
  tree_pin="${7:-$tree}"
  assert_rc "$expected" env "ORESLANG_BIN=$binary" "EXPECTED_COMPILER_SHA256=$hash" \
    "EXPECTED_ML_HEAD_SHA=$head_pin" "EXPECTED_LINEAR_BLOB=$linear_pin" \
    "EXPECTED_SMOKE_BLOB=$smoke_pin" "EXPECTED_ML_TREE_SHA=$tree_pin" sh "$script"
}
# No pins, no absolute executable, forged source or fixture, and wrong binary
# must all fail closed with UNCONFIGURED exit 78.
assert_rc 78 env ORESLANG_BIN='' sh "$script"
run_case 78 true "$sha_good" "$head" "$linear" "$smoke"
run_case 78 "$tmp/mock-accept" "$sha_good" "$(printf '%040d' 1)" "$linear" "$smoke"
run_case 78 "$tmp/mock-accept" "$sha_good" "$head" "$(printf '%040d' 1)" "$smoke"
run_case 78 "$tmp/mock-accept" "$sha_good" "$head" "$linear" "$(printf '%040d' 1)"
run_case 78 "$tmp/mock-accept" "$sha_no_output" "$head" "$linear" "$smoke"
# A valid HEAD must also point at an independently supplied expected tree.
run_case 78 "$tmp/mock-accept" "$sha_good" "$head" "$linear" "$smoke" "$(printf '%040d' 1)"
# Dirty tracked and untracked files must not be silently used for compilation.
printf 'tampered extra file\n' > "$tmp/work/tests/other.txt"
run_case 78 "$tmp/mock-accept" "$sha_good" "$head" "$linear" "$smoke"
printf 'dummy extra tracked file\n' > "$tmp/work/tests/other.txt"
printf 'untracked source\n' > "$tmp/work/src/rogue.ores"
run_case 78 "$tmp/mock-accept" "$sha_good" "$head" "$linear" "$smoke"
rm "$tmp/work/src/rogue.ores"
# An executable that exits zero but does not emit the exact marker is NOT a pass.
run_case 1 "$tmp/mock-no-output" "$sha_no_output" "$head" "$linear" "$smoke"
run_case 1 "$tmp/mock-noise" "$sha_noise" "$head" "$linear" "$smoke"
run_case 1 "$tmp/mock-extra" "$sha_extra" "$head" "$linear" "$smoke"
run_case 1 "$tmp/mock-nonzero" "$sha_nonzero" "$head" "$linear" "$smoke"
# A forged program cannot rewrite the verified source or fixture after preflight.
run_case 78 "$tmp/mock-source-mutate" "$sha_source_mutate" "$head" "$linear" "$smoke"
printf 'dummy affine kernel\n' > "$tmp/work/src/linear.ores"
run_case 78 "$tmp/mock-fixture-mutate" "$sha_fixture_mutate" "$head" "$linear" "$smoke"
printf 'dummy smoke fixture\n' > "$tmp/work/tests/smoke.ores"
# A forged program cannot create a new import source or change a different
# tracked file while still claiming the same reviewed source checkout.
run_case 78 "$tmp/mock-add-untracked" "$sha_add_untracked" "$head" "$linear" "$smoke"
rm "$tmp/work/src/rogue.ores"
run_case 78 "$tmp/mock-extra-mutate" "$sha_extra_mutate" "$head" "$linear" "$smoke"
printf 'dummy extra tracked file\n' > "$tmp/work/tests/other.txt"
# The guest must not move HEAD or commit altered tracked bytes while leaving
# a superficially clean checkout. Restore the test repository after each.
run_case 78 "$tmp/mock-empty-commit" "$sha_empty_commit" "$head" "$linear" "$smoke"
git -C "$tmp/work" reset --hard -q "$head"
run_case 78 "$tmp/mock-tree-commit" "$sha_tree_commit" "$head" "$linear" "$smoke"
git -C "$tmp/work" reset --hard -q "$head"
# Positive structural case deliberately uses a FAKE compiler: output+hash
# matches are NOT compiler attestation or a native guest execution proof.
run_case 0 "$tmp/mock-accept" "$sha_good" "$head" "$linear" "$smoke"
grep -q '^UNAUTHENTICATED_GUEST_OUTPUT_MATCH ' "$tmp/actual-stdout"
echo 'SOURCE_ONLY_SMOKE_VERIFIER_SELFTEST_PASS cases=20 native_guest=NOT_EXECUTED'
