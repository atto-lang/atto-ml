// HOST-ONLY source and independent integer calculus reference.
// This is not native actor execution and not automatic differentiation.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../examples/two_stage_actor_vjp_bootstrap.ores",import.meta.url),"utf8");

function pipeline(x,w0,w1,target) {
  const hidden=x*w0;
  const prediction=hidden*w1;
  const dy=2*(prediction-target);
  return {
    hidden,prediction,dy,
    hiddenGradient:dy*hidden,
    upstream:dy*w1,
    inputGradient:dy*w1*x
  };
}
function oneStep() {
  const stage0={state:{weight:2,version:1,savedInput:null,issued:false,applied:false}};
  const stage1={state:{weight:3,version:1,savedInput:null,issued:false,applied:false}};
  const x=3,target=16;
  stage0.state.savedInput=x;stage0.state.issued=true;
  const activation=x*stage0.state.weight;
  stage1.state.savedInput=activation;stage1.state.issued=true;
  const prediction=activation*stage1.state.weight;
  const grad=2*(prediction-target);
  const old=stage1.state.weight;
  const hiddenGrad=stage1.state.savedInput*grad;
  stage1.state.weight-=hiddenGrad;
  stage1.state.version+=1;stage1.state.applied=true;stage1.state.issued=false;
  const upstream=grad*old;
  const firstGrad=stage0.state.savedInput*upstream;
  stage0.state.weight-=firstGrad;
  stage0.state.version+=1;stage0.state.applied=true;stage0.state.issued=false;
  return {firstGrad,hiddenGrad,upstream,stage0,stage1,prediction};
}

test("all three actor classes declare exactly one state block",()=>{
  for(const name of ["InputWeight","HiddenWeight","SquaredLoss"]) {
    const start=source.indexOf("actor "+name);
    assert(start>=0);
    const end=source.indexOf("\nend",start);
    const region=source.slice(start,end);
    assert.equal((region.match(/\bstate\s*\{/g)||[]).length,1);
  }
  assert.doesNotMatch(source,/\bself\.state\b/);
  assert.doesNotMatch(source,/^\s*(?:pub\s+)?val\s+\w+\s+weight\s*;/m);
});
test("forward and backward are typed pipe messages not actor-method RPCs",()=>{
  assert.match(source,/define enum Signal as/);
  assert.match(source,/define transform pipe actor HiddenWeight/);
  assert.match(source,/ActorUtils\.pipe\(source, hidden, loss\)/);
  assert.match(source,/self\.forward\(Signal\.Data\(/);
  assert.match(source,/self\.backward\(Signal\.Gradient\(/);
  assert.match(source,/expect_pipeline\(source, 36\)/);
});
test("effect bounds and short RW loans cover all state access",()=>{
  assert.match(source,/receive\(Signals m\) with state\.read, state\.write/);
  assert.match(source,/receive_forward\(ForwardSignal m\): void with state\.read, state\.write/);
  assert.match(source,/receive_backward\(BackwardSignal m\): void with state\.read, state\.write/);
  assert.match(source,/using state\.write\(critical \|s\| ->/);
  assert.match(source,/using state\.read\(\|s\| ->/);
});
test("stage 1 uses old weight before publishing an update",()=>{
  const region=source.slice(source.indexOf("receive_backward("),source.indexOf("define actor SquaredLoss"));
  assert(region.indexOf("const old_weight = using state.read")>=0);
  assert(region.indexOf("const upstream = old_weight * dy")<region.indexOf("s.weight = s.weight - gradient"));
  assert(region.indexOf("s.weight = s.weight - gradient")<region.indexOf("self.backward(Signal.Gradient(upstream))"));
});
test("forward activation and gradients agree with independent scalar calculus",()=>{
  assert.deepEqual(pipeline(3,2,3,16),{
    hidden:6,prediction:18,dy:4,hiddenGradient:24,upstream:12,inputGradient:36
  });
});
test("numerical gradients agree with central differences over different weights",()=>{
  const eps=1e-5,x=3,target=16;
  const L=(a,b)=>{const error=x*a*b-target;return error*error;};
  for(const [a,b] of [[2,3],[1,4],[3,-1],[-2,2]]) {
    const g=pipeline(x,a,b,target);
    const d0=(L(a+eps,b)-L(a-eps,b))/(2*eps);
    const d1=(L(a,b+eps)-L(a,b-eps))/(2*eps);
    assert(Math.abs(d0-g.inputGradient)<1e-7);
    assert(Math.abs(d1-g.hiddenGradient)<1e-7);
  }
});
test("local step versions update exactly once and save immutable old weights",()=>{
  const actual=oneStep();
  assert.equal(actual.firstGrad,36);
  assert.equal(actual.hiddenGrad,24);
  assert.equal(actual.upstream,12);
  assert.equal(actual.stage0.state.version,2);
  assert.equal(actual.stage1.state.version,2);
  assert.equal(actual.stage0.state.weight,-34);
  assert.equal(actual.stage1.state.weight,-21);
  assert.equal(actual.stage0.state.issued,false);
  assert.equal(actual.stage1.state.issued,false);
});
test("user code contains only the one-message bootstrap and rejects duplicate update",()=>{
  assert.match(source,/s\.applied \|\| s\.version != 1/);
  assert.match(source,/s\.applied = true/);
  assert.match(source,/s\.issued = false/);
  assert.equal((source.match(/source\.send\(Signal\.Data\(/g)||[]).length,1);
  assert.doesNotMatch(source,/\bawait\b|\bself\.state\b/);
});
