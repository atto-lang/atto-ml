// Host-only manifest integrity. Actual compiler rejection must be demonstrated
// separately by Java/Graal and LLVM on pinned .ores source blobs.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const root=new URL("./fixtures/actor-state-ml-negatives/",import.meta.url);
const manifest=JSON.parse(readFileSync(new URL("cases.json",root),"utf8"));
test("six nontrivial compiler rejection gates remain explicitly unexecuted",()=>{
  assert.equal(manifest.contract,"oreslang.ml.actor-state.compiler-negative.v1");
  assert.equal(manifest.guest_execution,"NOT_EXECUTED");
  assert.equal(manifest.cases.length,6);
  const names=new Set();
  const classes=new Set();
  for(const item of manifest.cases){
    assert.equal(item.expected,"reject");
    assert.equal(item.guest_execution,"NOT_EXECUTED");
    assert(!names.has(item.fixture));
    assert(!classes.has(item.diagnostic_class));
    names.add(item.fixture);classes.add(item.diagnostic_class);
    assert.match(item.fixture,/^actor_ml_[a-z_]+_reject\.ores$/);
    const src=readFileSync(new URL(item.fixture,root),"utf8");
    assert.match(src,/define actor BadActor as/);
    assert.match(src,/state\s*\{/);
  }
});
