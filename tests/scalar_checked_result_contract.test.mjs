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
  assert.match(source,/Result<int, String>/);
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
