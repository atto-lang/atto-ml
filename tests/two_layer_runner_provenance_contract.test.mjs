// Host-only checks on the fail-closed native-guest admissions script.
// Does not run a compiler; no private source is copied to a carrier repo.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const script=readFileSync(new URL("../scripts/verify-two-layer-tensor-guest.sh",import.meta.url),"utf8");
test("requires four exact source heads and the runner executable hash",()=>{
  for(const key of [
    "ORESLANG_ML_EXPECTED_SHA","ORESLANG_DATA_EXPECTED_SHA",
    "ORESLANG_AUTOGRAD_EXPECTED_SHA","ORESLANG_TENSOR_EXPECTED_SHA",
    "ORESLANG_RUNNER_SHA256"
  ])assert(script.includes(key));
  assert.match(script,/if \[\[ -z "\$\{!required:-\}" \]\]/);
});
test("checks each repository revision, dirty files, and possible git worktrees",()=>{
  assert.match(script,/git -C "\$dir" rev-parse HEAD/);
  assert.match(script,/git -C "\$dir" status --porcelain --untracked-files=all/);
  assert.match(script,/\! -e "\$dir\/\.git"/);
  for(const name of [
    "oreslang-data-science","oreslang-autograd","oreslang-tensor"
  ])assert(script.includes(name));
});
test("checksum and required imports verified before launching guest",()=>{
  assert.match(script,/runner must be an absolute executable file/);
  assert.match(script,/sha256sum "\$runner"/);
  assert.match(script,/shasum -a 256 "\$runner"/);
  const check=script.indexOf('if [[ "$runner_hash" != "$ORESLANG_RUNNER_SHA256" ]]');
  const launch=script.indexOf('if ! "$@" "$root/examples/train_nullable_two_layer_autograd.ores"');
  assert(check>0&&launch>check);
  for(const source of [
    "ml_batch_bridge.ores","tape.ores","forward_checked.ores",
    "backward_checked.ores","row_gather_vjp.ores","row_batches.ores"
  ])assert(script.includes(source));
});
test("requires exact output rather than accepting arbitrary successful exit",()=>{
  assert.match(script,/ORES_TWO_LAYER_AUTOGRAD_PASS/);
  assert.match(script,/diff -u "\$temp\/expected" "\$temp\/stdout"/);
  assert.match(script,/ORESLANG_TWO_LAYER_GUEST=FAILED/);
  assert.match(script,/ORESLANG_TWO_LAYER_GUEST=OUTPUT_MISMATCH/);
  assert.match(script,/ORESLANG_TWO_LAYER_GUEST=PASS/);
});
