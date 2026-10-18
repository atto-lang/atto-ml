// Independent host-only checks; never treat as compiler execution.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../src/training/scalar_checked_result.ores",import.meta.url),"utf8");
const guest=readFileSync(new URL("../examples/typed_scalar_step_result.ores",import.meta.url),"utf8");
const bounded=n=>Number.isSafeInteger(n)&&n>=-10000&&n<=10000;
test("forward result separates legitimate negative success from failure",()=>{
  const value=(-2)*4;
  assert.equal(value,-8);
  assert.equal(bounded(value),true);
  assert.equal(bounded(10001),false);
  assert.match(source,/Result<int, ScalarTrainingError>/);
  assert.doesNotMatch(source,/Result<int, String>|Err\\("/);
  assert.match(guest,/import type ScalarTrainingError from/);
  assert.match(guest,/Result<int, ScalarTrainingError>/);
  assert.match(guest,/negative_success\.unwrap\(\) != -8/);
});
test("all four typed operations return errors separately from values",()=>{
  for(const n of ["checked_scalar_forward","checked_scalar_gradient",
    "checked_scalar_candidate","checked_scalar_retirement"]) {
    assert(source.includes("pub fnc "+n+"("));
    assert(guest.includes(n+"("));
  }
  assert.match(guest,/ORES_TYPED_SCALAR_RESULT_PASS/);
});
test("full gradient fan-in is mandatory before candidate or retirement",()=>{
  assert.match(source,/if receipts != 2/);
  assert.match(source,/return Err\("INCOMPLETE_FANIN"\)/);
  assert.match(source,/return Err\("VALID_CANDIDATE_MUST_COMMIT"\)/);
  assert.match(source,/return Ok\(version \+ 1\)/);
});
test("4096 seeded numeric cases stay inside the scalar safe domain",()=>{
  let seed=20261018;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
  for(let i=0;i<4096;i++){
    const w=random()%20001-10000;
    const x=random()%20001-10000;
    const upstream=random()%20001-10000;
    assert(Math.abs(w*x)<=100000000);
    assert(Math.abs(x*upstream)<=100000000);
    assert(Math.abs(w-x*upstream)<=100010000);
  }
});


test("zero receipts require zero gradient and one receipt caps first contribution",()=>{
  assert.match(source,/if receipts == 0 && running_gradient != 0/);
  assert.match(source,/return Err\("UNBACKED_ACCUMULATOR"\)/);
  assert.match(source,/if receipts == 1/);
  assert.match(source,/running_gradient < -100000000 \|\| running_gradient > 100000000/);
  assert.match(guest,/checked_scalar_gradient\(1, 1, 1, 0\)\.is_err\(\)/);
  assert.match(guest,/checked_scalar_gradient\(1, 1, 100000001, 1\)\.is_err\(\)/);
});
test("receipt-admissibility boundaries are mathematically complete for one contribution",()=>{
  const canStage=(receipts,current,x,upstream)=>{
    if(receipts<0||receipts>=2)return false;
    if(!bounded(x)||!bounded(upstream))return false;
    if(receipts===0&&current!==0)return false;
    if(receipts===1&&Math.abs(current)>100000000)return false;
    return Math.abs(current+x*upstream)<=200000000;
  };
  assert.equal(canStage(0,0,10000,10000),true);
  assert.equal(canStage(0,1,1,1),false);
  assert.equal(canStage(1,100000000,10000,10000),true);
  assert.equal(canStage(1,100000001,1,1),false);
  assert.equal(canStage(2,0,1,1),false);
});

test("all pure helpers return a single closed nominal error domain",()=>{
  const expected=["INVALID_INPUT","BAD_RECEIPT_COUNT","INVALID_GRADIENT","INVALID_ACCUMULATOR","UNBACKED_ACCUMULATOR","ACCUMULATION_BOUNDS","INVALID_STATE","INCOMPLETE_FANIN","COUNTER_EXHAUSTED","UNSAFE_CANDIDATE","VALID_CANDIDATE_MUST_COMMIT"];
  const body=source.match(/define enum ScalarTrainingError as\\n([\\s\\S]*?)\\nend/);
  assert(body,"missing nominal pure helper enum");
  const declared=[...body[1].matchAll(/^\\s+([A-Z_]+) = (\\d+)$/gm)];
  assert.deepEqual(declared.map(m=>m[1]),expected);
  assert.deepEqual(declared.map(m=>Number(m[2])),expected.map((_,i)=>i+1));
  const used=[...source.matchAll(/Err\\(ScalarTrainingError\\.([A-Z_]+)\\(\\)\\)/g)]
    .map(m=>m[1]);
  assert(used.length>=11);
  assert(used.every(v=>expected.includes(v)));
  for(const op of ["checked_scalar_forward","checked_scalar_gradient",
      "checked_scalar_candidate","checked_scalar_retirement"]) {
    assert.match(source,new RegExp("pub fnc "+op+"\\\\([\\\\s\\\\S]*?\\\\): Result<int, ScalarTrainingError>"));
  }
  assert.doesNotMatch(source,/Err\\("|Result<int, String>/);
  assert.doesNotMatch(guest,/Result<int, String>/);
});
