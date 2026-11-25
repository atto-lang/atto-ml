# Actions admission blocker — KAN native acceptance runbook

Status: **BLOCKED BEFORE RUNNER START**, not a numerical or compiler regression. Do not merge PR #19 on this basis.

## Exact evidence — 2026-10-25

Repository: `ores-truffle-oreslang/oreslang-ml` (private; immutable GitHub repository ID `1423913534`).

- New **source-local no-checkout** canary workflow at `.github/workflows/source-org-runner-admission.yml`: [run 40264674829](https://github.com/ores-truffle-oreslang/oreslang-ml/actions/runs/40264674829), job `121384331381`, concluded failure with `steps=null`. This job uses `permissions: {}`, ubuntu-latest and a single shell `printf`, no checkout and no third-party actions.
- Same source PR head `bd0677ff71c2a81719654e43c6774fe97a547b6a` also produced [numerical 40264674803](https://github.com/ores-truffle-oreslang/oreslang-ml/actions/runs/40264674803) and [contract 40264674760](https://github.com/ores-truffle-oreslang/oreslang-ml/actions/runs/40264674760): both failed before steps.
- Independent **source-free** canary in the review-only alternate organization: [run 40253849945](https://github.com/ores-dummy-org-oreslang-stack/ores-dummy-org-oreslang-stack-infra/actions/runs/40253849945) on [PR #58](https://github.com/ores-dummy-org-oreslang-stack/ores-dummy-org-oreslang-stack-infra/pull/58), also failed with `steps=null`.

These symptoms do **not** reveal the root cause. In particular, they do not establish depleted Actions minutes, a failed payment, a policy block or a GitHub-side provisioning incident.

## Source-organization administrator checklist

1. At the organization **Settings → Billing and licensing → Budgets and alerts**, inspect the Actions budget at the enterprise, organization, and repository levels, including any overlapping metered limits and **Stop usage when budget limit is reached**. Review billing hold/payment notices and recent usage. Do not silently increase spending.
2. At organization and repository **Settings → Actions → General**, confirm workflow execution is allowed for private repositories and that job admission is not awaiting approval. Verify the settings inherited from enterprise policies.
3. At repository **Settings → Actions → Runners**, check for an authorized runner, its state, labels, group access, and online availability. Even if no self-hosted runner exists, the basic ubuntu-latest canary above should ordinarily be schedulable with an eligible GitHub-hosted allowance; confirm any rejection reason via GitHub support.
4. Open the failed canary run and inspect the job annotation/check summary for a pre-run scheduler or billing message. The connector exposes no usable job steps or logs. If no UI-level reason exists, contact GitHub Support with the exact three source run IDs, the alternate-org source-free canary, repository numeric ID and the time window. Ask for the **workflow/job admission reason**, not a compiler diagnosis.
5. Re-run **one** no-checkout source-org canary after the setting/support change. Require a real job with nonempty `steps` and the exact `SOURCE_ORG_RUNNER_STEP_EXECUTED=YES` marker before scheduling compiler work. Stop repeatedly retrying identical zero-step runs.

References:
- [GitHub Docs: budgets and alerts](https://docs.github.com/en/billing/concepts/budgets-and-alerts)
- [GitHub Docs: setting up budgets](https://docs.github.com/en/billing/how-tos/set-up-budgets)
- [GitHub Docs: self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
- [GitHub Docs: self-hosted runner security](https://docs.github.com/en/actions/reference/security/secure-use)

## Native compiler acceptance after admission

Do not use arbitrary public binaries, repo-selected SHA-256 strings, or source-PR workflow jobs on privileged persistent machines. An authorized administrator must provision a trusted **isolated, disposable** runner or preapproved trusted compiler environment, bind it only to this source organization and reviewed code, and record both the source commit and independently approved compiler SHA-256.

Then run the existing `scripts/verify-kan-native-cpu-suite.sh` against **each** actual Java/GraalVM and LLVM compiler (separate compiler pins), collecting three exact native guest receipts and runner step logs for each backend. PR #19's host-only numerical workflow and synthetic receipt tests are not substitutes.

## Security boundary

The authoritative source-governance destination allowlist at `ores-truffle-oreslang/oreslang-source.java:.github/private-ci-approved-repositories.txt` has **no approved numeric destinations**. Cross-organization Git-tree mirroring through the GitHub connector **still transfers private code**; availability of minutes or a source-free successful canary does not approve transfer. No private source was copied to the alternate organization.

Track this unblocker in [oreslang-ml issue #20](https://github.com/ores-truffle-oreslang/oreslang-ml/issues/20).


## Reproducible read-only diagnostics

An organization administrator with an appropriately scoped `gh` CLI login and `jq` can run, from the source repo or another trusted workstation:

```bash
bash scripts/diagnose-kan-actions-admission.sh ores-truffle-oreslang/oreslang-ml 40264864016
bash scripts/diagnose-kan-actions-admission.sh ores-dummy-org-oreslang-stack/ores-dummy-org-oreslang-stack-infra 40253849945
```

This helper performs only GitHub REST **GET** requests for one run, its jobs, timing, and (when permitted) repo Actions enablement and self-hosted runner counts. No token is printed or stored; the trusted `gh` CLI manages authentication. Missing admin access prints `UNAVAILABLE`, never a false policy verdict. `scripts/test-kan-actions-admission-diagnostics.sh` runs thirteen source-free mocked regression cases and never contacts GitHub.

Additional observed raw GitHub REST job fields:
- Source-local no-checkout canary `121384937647`: `runner_id=0`, empty runner name, `steps=[]`; 0 billed Ubuntu milliseconds; overall run duration 3,000 ms.
- Alternate-org source-free canary `121351350771`: `runner_id=0`, empty runner name, `steps=[]`; 0 billed Ubuntu milliseconds; overall run duration 4,000 ms.

**Interpretation:** Neither test was assigned a runner, but these reads alone cannot distinguish exhausted budgets, repository policy, account restrictions, or scheduler provisioning failures. GitHub billing documents note that overlapping hard-stop budgets can prevent hosted runner usage: https://docs.github.com/en/billing/how-tos/set-up-budgets.


### Diagnostic classification hardening (October 25)

The diagnostic now treats a pending/queued run as `RUN_NOT_COMPLETED`, not as evidence of failed admission; never interprets a truncated first page of job results as complete; and counts steps with `status=completed` rather than merely listed steps. It also preserves a real repository Actions policy `enabled=false` (previously lost when `jq //` substituted `UNKNOWN`) and reports listed rather than whole-organization self-hosted runner totals when pages may be incomplete. Mock coverage increased from 8 to **13/13 passing**; the full seven-suite host V8 math check is **67/67 passing**, independently of Actions. These counts do **not** establish native Java/Graal or LLVM compiler execution.

At [PR #19](https://github.com/ores-truffle-oreslang/oreslang-ml/pull/19), the exact-head source-org no-checkout canary and the two host workflows continue to fail without any step execution. No private source has been copied to the review-only alternate organization; no destination has been approved by the source governance allowlist.

## Active repository migration checkpoint

For current AttoLang work, use `atto-lang/atto-ml` (repository ID `1448720923`); older run IDs and private repository references above are historical. Do not copy private dependencies or use cross-organization credentials. Public-source diagnostics do not establish private-export authority or native ML execution.
