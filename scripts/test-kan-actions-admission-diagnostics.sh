#!/usr/bin/env bash
# Source-free mocks: verify the read-only admission diagnostic has no writes.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"
cat > "$tmp/bin/gh" <<'GH'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == api && "$2" == --method && "$3" == GET ]] || exit 10
endpoint="${@: -1}"
case "$endpoint" in
  */actions/runs/40264864016)
    if [[ "${MOCK_SCENARIO:-}" == bad_id ]]; then printf '{"id":123,"status":"completed","conclusion":"failure","head_sha":"abc"}\n'; else
      printf '{"id":40264864016,"status":"completed","conclusion":"failure","head_sha":"abc"}\n'; fi ;;
  */actions/runs/40264864016/jobs\?per_page=100)
    case "${MOCK_SCENARIO:-}" in
      stepful) printf '{"jobs":[{"runner_id":17,"steps":[{"name":"run"}],"conclusion":"success"}]}\n' ;;
      empty) printf '{"jobs":[]}\n' ;;
      *) printf '{"jobs":[{"runner_id":0,"steps":[],"conclusion":"failure"}]}\n' ;;
    esac ;;
  */actions/runs/40264864016/timing) printf '{"billable":{"UBUNTU":{"total_ms":0}}}\n' ;;
  */actions/permissions) [[ "${MOCK_SCENARIO:-}" == stepful ]] && printf '{"enabled":true}\n' || exit 4 ;;
  */actions/runners\?per_page=100) [[ "${MOCK_SCENARIO:-}" == stepful ]] && printf '{"runners":[{"status":"online"},{"status":"offline"}]}\n' || exit 4 ;;
  *) exit 4 ;;
esac
GH
chmod +x "$tmp/bin/gh"
count=0
check() {
  local rc_expected="$1" substring="$2" scenario="$3" ; shift 3
  local output rc=0
  output="$(env PATH="$tmp/bin:$PATH" MOCK_SCENARIO="$scenario" bash "$root/diagnose-kan-actions-admission.sh" "$@" 2>&1)" || rc=$?
  if [[ "$rc" != "$rc_expected" || "$output" != *"$substring"* ]]; then
    printf 'FAIL expected %s %s, got %s %s\n' "$rc_expected" "$substring" "$rc" "$output" >&2
    exit 1
  fi
  ((count+=1))
}
check 2 'USAGE' pass 'bad/repo/escape' 40264864016
check 2 'USAGE' pass 'ores-truffle-oreslang/oreslang-ml' NaN
check 2 'API shape or run ID mismatch' bad_id 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'ADMISSION=NO_RUNNER_ASSIGNED' blocked 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'REPOSITORY_ACTIONS_ENABLED=UNAVAILABLE' blocked 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'ADMISSION=STEPFUL_EXECUTION' stepful 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'SELF_HOSTED_RUNNERS_ONLINE=1' stepful 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'ADMISSION=INDETERMINATE' empty 'ores-truffle-oreslang/oreslang-ml' 40264864016
printf 'DIAGNOSTIC_MOCK_TESTS=%d PASS\n' "$count"
