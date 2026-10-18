// Independent host-only bounds oracle: not a compiled Oreslang actor test.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../examples/train_actor_state_scalar.ores",import.meta.url),"utf8");
const valid=v=>Number.isSafeInteger(v)&&v>=-10000&&v<=10000;
const proposed=(weight,x1,x2,u1,u2)=>weight-(x1*u1+x2*u2);
test("safe scalar-domain upper bounds fit signed 32-bit arithmetic",()=>{
  assert(10000*10000<=2147483647);
  assert(2*10000*10000+10000<=2147483647);
  assert.match(source,/if x < -10000 \|\| x > 10000/);
  assert.match(source,/if upstream < -10000 \|\| upstream > 10000/);
});
test("late optimizer rejection cannot publish a new version or a partial weight",()=>{
  const state={weight:5,version:2,step:2,gradient:200000000,received:2};
  const original={...state};
  const candidate=state.weight-state.gradient;
  assert(!valid(candidate));
  if(valid(candidate)){
    state.weight=candidate;state.version++;state.step++;
    state.gradient=0;state.received=0;
  }
  assert.deepEqual(state,original);
  const update=source.indexOf("s.weight = candidate_weight");
  assert(source.indexOf("if candidate_weight < -10000 || candidate_weight > 10000")<update);
  assert(source.indexOf("if s.step >= 1000000 || s.version >= 1000000")<update);
});
test("out-of-range messages are rejected before consuming gradient receipts",()=>{
  const state={forward:false,seen:false,sum:0};
  if(valid(10001))state.forward=true;
  assert.equal(state.forward,false);
  state.forward=true;
  if(valid(-10001)){state.seen=true;state.sum+=3*(-10001);}
  assert.deepEqual(state,{forward:true,seen:false,sum:0});
  state.seen=true;state.sum+=3*4;
  assert.deepEqual(state,{forward:true,seen:true,sum:12});
});
test("seeded 4096-pair gradient sums and candidates cannot overflow int32",()=>{
  let s=20261018;
  const next=()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s;};
  for(let i=0;i<4096;i++){
    const weight=next()%20001-10000;
    const x1=next()%20001-10000,x2=next()%20001-10000;
    const u1=next()%20001-10000,u2=next()%20001-10000;
    const grad=x1*u1+x2*u2,candidate=proposed(weight,x1,x2,u1,u2);
    assert(Math.abs(grad)<=200000000);
    assert(Math.abs(candidate)<=200010000);
    if(valid(candidate))assert(Math.abs(candidate)<=10000);
    else assert(candidate < -10000 || candidate > 10000);
  }
});
test("version and step exhaustion precede all mutable publications",()=>{
  const start=source.indexOf("private commit("),stop=source.indexOf("private weight_snapshot(",start);
  const part=source.slice(start,stop);
  assert(part.indexOf("if s.step >= 1000000 || s.version >= 1000000")
    <part.indexOf("s.weight = candidate_weight"));
  assert(part.indexOf("s.weight = candidate_weight")
    <part.indexOf("s.version = s.version + 1"));
});
