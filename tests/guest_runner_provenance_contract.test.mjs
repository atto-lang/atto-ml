// Host-only validation of fail-closed runner provenance contracts.
// This does NOT execute a private compiler or prove any source tree is safe.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const script=readFileSync(new URL("../scripts/verify-data-autograd-guest.sh",import.meta.url),"utf8");
test("guest runner requires explicit identities for ML, data, autograd and runner",()=>{
  for(const key of [
    "ORESLANG_ML_EXPECTED_SHA","ORESLANG_DATA_EXPECTED_SHA",
    "ORESLANG_AUTOGRAD_EXPECTED_SHA","ORESLANG_RUNNER_SHA256"
  ]) assert(script.includes(key),key);
  assert.match(script,/if \[\[ -z "\$\{!required:-\}" \]\]/);
});
test("refuses dirty or mismatched independent source checkouts",()=>{
  assert.match(script,/git -C "\$dir" rev-parse HEAD/);
  assert.match(script,/git -C "\$dir" status --porcelain --untracked-files=all/);
  assert.match(script,/source revision mismatch/);
  assert.match(script,/dirty source worktree/);
  assert.match(script,/\! -d "\$dir\/\.git"/);
});
test("runner executable checksum required and checked before guest invocation",()=>{
  assert.match(script,/sha256sum "\$runner"/);
  assert.match(script,/shasum -a 256 "\$runner"/);
  assert.match(script,/runner executable hash mismatch/);
  const pin=script.indexOf('if [[ "$runner_hash" != "$ORESLANG_RUNNER_SHA256" ]]');
  const invoke=script.indexOf('if ! "$@" "$root/examples/train_nullable_data_with_autograd.ores"');
  assert(pin>0&&invoke>pin);
});
test("guest verification requires single exact stdout marker",()=>{
  assert.match(script,/ORES_DATA_AUTOGRAD_TRAINING_PASS/);
  assert.match(script,/diff -u "\$temp\/expected" "\$temp\/stdout"/);
  assert.match(script,/ORESLANG_DATA_AUTOGRAD_GUEST=PASS/);
  assert.match(script,/exit 78/);
});
