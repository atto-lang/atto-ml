#!/usr/bin/env bash
# Strict, source-free read-only admission classifier mock cases.
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
    case "${MOCK_SCENARIO:-}" in
      bad_id) printf '{"id":123,"status":"completed","conclusion":"failure","head_sha":"abc"}\n' ;;
      queued|partial) printf '{"id":40264864016,"status":"queued","conclusion":null,"head_sha":"abc"}\n' ;;
      *) printf '{"id":40264864016,"status":"completed","conclusion":"failure","head_sha":"abc"}\n' ;;
    esac ;;
  */actions/runs/40264864016/jobs\?per_page=100)
    case "${MOCK_SCENARIO:-}" in
      stepful) printf '{"total_count":1,"jobs":[{"runner_id":17,"steps":[{"name":"run","status":"completed","conclusion":"success"}],"conclusion":"success"}]}\n' ;;
      steps_pending) printf '{"total_count":1,"jobs":[{"runner_id":17,"steps":[{"name":"run","status":"queued"}]}]}\n' ;;
      empty) printf '{"total_count":0,"jobs":[]}\n' ;;
      partial) printf '{"total_count":101,"jobs":[{"runner_id":0,"steps":[]}]}\n' ;;
      malformed) printf '{"jobs":[{"runner_id":0,"steps":[]}]}\n' ;;
      *) printf '{"total_count":1,"jobs":[{"runner_id":0,"steps":[],"conclusion":"failure"}]}\n' ;;
    esac ;;
  */actions/runs/40264864016/timing) printf '{"billable":{"UBUNTU":{"total_ms":0}}}\n' ;;
  */actions/permissions)
    case "${MOCK_SCENARIO:-}" in
      stepful) printf '{"enabled":true}\n' ;;
      disabled) printf '{"enabled":false}\n' ;;
      *) exit 4 ;;
    esac ;;
  */actions/runners\?per_page=100)
    if [[ "${MOCK_SCENARIO:-}" == stepful ]]; then
      printf '{"total_count":2,"runners":[{"status":"online"},{"status":"offline"}]}\n'
    else exit 4; fi ;;
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
check 2 'API shape or run ID mismatch' malformed 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'ADMISSION=NO_RUNNER_ASSIGNED' blocked 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'REPOSITORY_ACTIONS_ENABLED=UNAVAILABLE' blocked 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'ADMISSION=STEPFUL_EXECUTION' stepful 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'SELF_HOSTED_RUNNERS_ONLINE_LISTED=1' stepful 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'REPOSITORY_ACTIONS_ENABLED=false' disabled 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'ADMISSION=INDETERMINATE' empty 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'ADMISSION=RUN_NOT_COMPLETED' queued 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'ADMISSION=INDETERMINATE_PARTIAL_JOB_PAGE' partial 'ores-truffle-oreslang/oreslang-ml' 40264864016
check 0 'ADMISSION=INDETERMINATE' steps_pending 'ores-truffle-oreslang/oreslang-ml' 40264864016
printf 'DIAGNOSTIC_MOCK_TESTS=%d PASS\n' "$count"
