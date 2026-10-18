#!/usr/bin/env bash
# Authoritative only when supplied an approved local compiler executable and
# exact clean ML source tree. Does not fetch private code or mint CI evidence.
set -euo pipefail
if (( $# < 1 )); then
  echo "usage: $0 /absolute/path/to/pinned/oreslang [runner arguments...]" >&2
  exit 64
fi
if [[ -z "${ORESLANG_ML_EXPECTED_SHA:-}" || -z "${ORESLANG_RUNNER_SHA256:-}" ]]; then
  echo "missing ORESLANG_ML_EXPECTED_SHA or ORESLANG_RUNNER_SHA256" >&2
  exit 64
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -e "$root/.git" ]] || { echo "missing ML Git checkout" >&2; exit 78; }
test "$(git -C "$root" rev-parse HEAD)" = "$ORESLANG_ML_EXPECTED_SHA" || {
  echo "ML source revision mismatch" >&2; exit 78;
}
test -z "$(git -C "$root" status --porcelain --untracked-files=all)" || {
  echo "dirty ML checkout" >&2; exit 78;
}
runner="$1"
if [[ "$runner" != /* || ! -f "$runner" || ! -x "$runner" ]]; then
  echo "runner must be an absolute executable file" >&2
  exit 64
fi
if command -v sha256sum >/dev/null 2>&1; then
  digest="$(sha256sum "$runner")"; digest="${digest%% *}"
elif command -v shasum >/dev/null 2>&1; then
  digest="$(shasum -a 256 "$runner")"; digest="${digest%% *}"
else
  echo "SHA-256 utility unavailable" >&2
  exit 78
fi
test "$digest" = "$ORESLANG_RUNNER_SHA256" || {
  echo "runner executable hash mismatch" >&2; exit 78;
}
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf 'ORES_TYPED_SCALAR_RESULT_PASS\n' > "$tmp/expected"
if ! "$@" "$root/examples/typed_scalar_step_result.ores" > "$tmp/stdout" 2> "$tmp/stderr"; then
  cat "$tmp/stderr" >&2
  echo "ORES_TYPED_SCALAR_GUEST=FAILED" >&2
  exit 1
fi
if ! diff -u "$tmp/expected" "$tmp/stdout"; then
  cat "$tmp/stderr" >&2
  echo "ORES_TYPED_SCALAR_GUEST=OUTPUT_MISMATCH" >&2
  exit 1
fi
echo "ORES_TYPED_SCALAR_GUEST=PASS"
