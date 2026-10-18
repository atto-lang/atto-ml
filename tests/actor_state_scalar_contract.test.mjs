// Host-only source spelling and independent scalar-gradient oracle tests.
// These do not execute the .ores fixture and are not compiler acceptance.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const file = new URL("../examples/train_actor_state_scalar.ores", import.meta.url);
const source = readFileSync(file, "utf8");

test("single state schema; no actor field storage or self.state bypass", () => {
  assert.match(source,/define actor VersionedScalarStage as/);
  assert.equal((source.match(/\bstate\s*\{/g)||[]).length,1);
  assert.doesNotMatch(source.replace(/\/\/[^\n]*/g,""),/\bself\.state\b/);
  assert.doesNotMatch(source,/^\s*(val|let)\s+(?:Tensor|Linear|Optimizer|Gradient)\s+/m);
});

test("with effect bounds are independent from lexical using authority", () => {
  assert.match(source,/forward\(int x\): int with state\.read/);
  assert.match(source,/using state\.read\(\|s\| -> s\.weight\)/);
  assert.match(source,/accumulate\([^)]*\): int\s+with state\.read, state\.write/);
  assert.match(source,/using state\.write\(critical \|s\| -> \{/);
  assert.match(source,/commit\([^)]*\): int with state\.read, state\.write/);
  assert.match(source,/self\.handle_state\(critical write \|s\| -> \{/);
  assert.match(source,/version\(\): int with state\.read/);
  assert.match(source,/self\.handle_state\(critical read \|s\| -> s\.version\)/);
  assert.doesNotMatch(source,/\bfnc\s+\w+\([^)]*\)\s+using\s+state/);
});

test("write callback checks weight version and gradient count before publishing", () => {
  const start=source.indexOf("pub commit("),end=source.indexOf("pub version()",start);
  const commit=source.slice(start,end);
  const guard=commit.indexOf("s.version != expected_version");
  const full=commit.indexOf("s.pending_gradients != 2");
  const update=commit.indexOf("s.weight = s.weight - s.accumulated_gradient");
  const bump=commit.indexOf("s.version = s.version + 1");
  assert(guard >= 0 && full > guard && update > full && bump > update);
  assert.doesNotMatch(commit,/\bawait\b|\bselect\b|rt cooperate/);
});

test("scalar VJP and versioned update independent oracle", () => {
  const w=10,inputs=[2,3],upstream=[1,1],version=1;
  assert.deepEqual(inputs.map(x=>w*x),[20,30]);
  const contribution=inputs.map((x,i)=>x*upstream[i]);
  assert.deepEqual(contribution,[2,3]);
  const sum=contribution.reduce((a,b)=>a+b,0);
  assert.equal(sum,5);
  const next=w-sum;
  assert.equal(next,5);
  assert.equal(version+1,2);
  assert.equal(next*2,10);
  assert.equal((1===version+1),false); // reject stale version-1 gradient
});
test("scalar weight VJP agrees with finite differences for many values", () => {
  for(const w of [-9,0,1,3,20])for(const x of [-4,0,1,2,8])for(const upstream of [-3,0,1,5]){
    const f=(v)=>upstream*(v*x);
    const finite=(f(w+1)-f(w-1))/2;
    assert(finite === x*upstream); // signed zero is not a gradient mismatch
  }
});
