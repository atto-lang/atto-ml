#!/usr/bin/env bash
# Usage: bash scripts/verify-actor-state-guest.sh COMMAND [ARG...]
# COMMAND must be a locally available, already built Oreslang guest runner,
# accepting the .ores source path as its final argument.
# This verifier intentionally does NOT supply a compiler, fetch private code,
# skip compilation, or turn a host-only oracle into guest-execution evidence.
set -euo pipefail
if (( $# == 0 )); then
  echo "usage: $0 GUEST_RUNNER [ARG...]" >&2
  exit 64
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
printf '%s\n' "guest_execution=ATTEMPTED runner=$1" >&2
"$@" "$root/examples/train_actor_state_scalar.ores" > "$temp/guest.stdout" 2> "$temp/guest.stderr" || {
  cat "$temp/guest.stderr" >&2
  echo "guest execution failed; NOT_VERIFIED" >&2
  exit 1
}
diff -u "$root/tests/golden/actor_state_scalar.stdout" "$temp/guest.stdout" || {
  cat "$temp/guest.stderr" >&2
  echo "guest stdout mismatch; NOT_VERIFIED" >&2
  exit 1
}
echo "actor scalar guest-output conformance PASS (review exact compiler revision separately)"
