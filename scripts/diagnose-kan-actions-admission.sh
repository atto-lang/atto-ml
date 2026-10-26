#!/usr/bin/env bash
# Read-only GitHub Actions admission evidence. This NEVER invokes runners,
# changes budgets, or transfers private source to another repository.
set -euo pipefail
repo="${1:-ores-truffle-oreslang/oreslang-ml}"
run_id="${2:-}"
if [[ ! "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ || ! "$run_id" =~ ^[1-9][0-9]*$ ]]; then
  echo 'USAGE: diagnose-kan-actions-admission.sh owner/repo numeric-run-id' >&2
  exit 2
fi
for binary in gh jq; do
  command -v "$binary" >/dev/null 2>&1 || {
    printf 'NOT_EXECUTED: missing required %s command\n' "$binary" >&2
    exit 2
  }
done
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
path="repos/$repo/actions/runs/$run_id"
get() {
  gh api --method GET -H 'Accept: application/vnd.github+json' "$1" 2> "$tmp/error"
}
if ! get "$path" > "$tmp/run"; then
  echo 'NOT_EXECUTED: Actions run API read denied or unavailable' >&2
  exit 2
fi
if ! get "$path/jobs?per_page=100" > "$tmp/jobs"; then
  echo 'NOT_EXECUTED: Actions jobs API read denied or unavailable' >&2
  exit 2
fi
if ! jq -e --argjson n "$run_id" '.id == $n' "$tmp/run" >/dev/null || ! jq -e '(.jobs|type) == "array"' "$tmp/jobs" >/dev/null; then
  echo 'NOT_EXECUTED: Actions API shape or run ID mismatch' >&2
  exit 2
fi
if get "$path/timing" > "$tmp/timing"; then
  ubuntu_ms="$(jq -r '.billable.UBUNTU.total_ms // "UNAVAILABLE"' "$tmp/timing")"
else
  ubuntu_ms=UNAVAILABLE
fi
read -r state conclusion head_sha < <(jq -r '[.status // "unknown",.conclusion // "unknown",.head_sha // "unknown"]|@tsv' "$tmp/run")
read -r jobs assigned steps < <(jq -r '[
  (.jobs|length),
  ([.jobs[]|select((.runner_id // 0) > 0)]|length),
  ([.jobs[]|(.steps // [])|length]|add // 0)
]|@tsv' "$tmp/jobs")
printf 'RUN repo=%s id=%s head=%s status=%s conclusion=%s\n' "$repo" "$run_id" "$head_sha" "$state" "$conclusion"
printf 'JOBS count=%s runner_assigned=%s executed_steps=%s billable_ubuntu_ms=%s\n' "$jobs" "$assigned" "$steps" "$ubuntu_ms"
if [[ "$jobs" -gt 0 && "$assigned" -eq 0 && "$steps" -eq 0 ]]; then
  echo 'ADMISSION=NO_RUNNER_ASSIGNED; ROOT_CAUSE=UNKNOWN'
elif [[ "$steps" -gt 0 ]]; then
  echo 'ADMISSION=STEPFUL_EXECUTION; NATIVE_GUEST=NOT_ESTABLISHED'
else
  echo 'ADMISSION=INDETERMINATE; ROOT_CAUSE=UNKNOWN'
fi
# Administrator-only reads. Absence of access is not a verdict on policies.
if get "repos/$repo/actions/permissions" > "$tmp/permissions"; then
  enabled="$(jq -r '.enabled // "UNKNOWN"' "$tmp/permissions")"
  echo "REPOSITORY_ACTIONS_ENABLED=$enabled"
else
  echo 'REPOSITORY_ACTIONS_ENABLED=UNAVAILABLE'
fi
if get "repos/$repo/actions/runners?per_page=100" > "$tmp/runners"; then
  online="$(jq -r '[.runners[]?|select(.status == "online")]|length' "$tmp/runners")"
  echo "SELF_HOSTED_RUNNERS_ONLINE=$online"
else
  echo 'SELF_HOSTED_RUNNERS_ONLINE=UNAVAILABLE'
fi
echo 'ACTIONS_BILLING_CAP_AND_ADMISSION_REASON=ADMIN_REVIEW_REQUIRED'
