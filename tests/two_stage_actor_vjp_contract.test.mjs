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
  assert(region.includes("using state.read(|s| ->"));
  assert(region.includes("(s.saved_input, s.weight, s.version, s.issued, s.applied)"));
  assert(region.includes("const (input, old_weight, observed_version, was_issued, was_applied)"));
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
  assert.match(source,/s\.applied \|\| s\.version != observed_version/);
  assert.match(source,/s\.applied = true/);
  assert.match(source,/s\.issued = false/);
  assert.equal((source.match(/source\.send\(Signal\.Data\(/g)||[]).length,1);
  assert.doesNotMatch(source,/\bawait\b|\bself\.state\b/);
});

test("a single RW read loan gives a consistent forward-context snapshot",()=>{
  const backward=source.slice(source.indexOf("receive_backward(ForwardSignal")>=0
      ? source.indexOf("receive_backward(ForwardSignal")
      : source.indexOf("receive_backward(BackwardSignal"));
  const relevant=backward.slice(0,backward.indexOf("define actor SquaredLoss"));
  const readLoans=[...relevant.matchAll(/using state\.read\(/g)].length;
  assert.equal(readLoans,1);
  assert.match(relevant,/observed_version != 1/);
  assert.match(relevant,/s\.version != observed_version/);
  assert.match(relevant,/was_issued \|\| was_applied/);
});
test("mixed read generations are rejected, whereas atomic snapshots are consistent",()=>{
  // Independent concurrency interleaving: a split read can return old x
  // with new w after a competing publication. One read-cohort snapshot cannot.
  const older={input:6,weight:3,version:1};
  const newer={input:7,weight:4,version:2};
  const inconsistent={input:older.input,weight:newer.weight,version:newer.version};
  assert.equal(inconsistent.input,6);
  assert.equal(inconsistent.weight,4);
  assert.deepEqual(
      Object.freeze({...older}),
      {input:6,weight:3,version:1});
  assert.deepEqual(
      Object.freeze({...newer}),
      {input:7,weight:4,version:2});
  assert.notDeepEqual(inconsistent,older);
  assert.notDeepEqual(inconsistent,newer);
});


test("each trainable stage rejects duplicate forwards before mutating state",()=>{
  const first=source.slice(source.indexOf("define actor InputWeight"),source.indexOf("define transform pipe actor HiddenWeight"));
  const second=source.slice(source.indexOf("define transform pipe actor HiddenWeight"),source.indexOf("define actor SquaredLoss"));
  for(const region of [first,second]){
    assert.match(region,/const \(accepted, w\) = using state\.write\(critical/);
    assert.match(region,/if s\.issued \|\| s\.applied \|\| s\.version != 1/);
    assert.match(region,/return \(false, 0\)/);
    assert.match(region,/return \(true, s\.weight\)/);
    assert.match(region,/if accepted; do\s+self\.forward\(Signal\.Data/);
  }
});
test("terminal loss forward dedupe returns before backward message emission",()=>{
  const tail=source.slice(source.indexOf("define actor SquaredLoss"));
  assert.match(tail,/const \(accepted, target\) = using state\.write\(critical/);
  assert.match(tail,/if s\.seen_forward; do\s+return \(false, 0\)/);
  assert.match(tail,/if !accepted; do\s+return -5/);
  assert(tail.indexOf("if !accepted; do") < tail.indexOf("self.backward(Signal.Gradient(gradient))"));
});


test("all three actor boundaries reject extreme integer messages before arithmetic",()=>{
  const first=source.slice(source.indexOf("define actor InputWeight"),source.indexOf("define transform pipe actor HiddenWeight"));
  const middle=source.slice(source.indexOf("define transform pipe actor HiddenWeight"),source.indexOf("define actor SquaredLoss"));
  const last=source.slice(source.indexOf("define actor SquaredLoss"));
  for(const [region,variable,bound] of [
    [first,"x",1000],[first,"dy",4096],[middle,"activation",1000],
    [middle,"dy",4096],[last,"prediction",1000]
  ]){
    const clause="if "+variable+" < -"+bound+" || "+variable+" > "+bound+"; do";
    assert(region.includes(clause),"missing fail-closed numeric bound: "+clause);
    const start=region.indexOf("on Signal."+(variable==="dy"?"Gradient":"Data"));
    const guarded=region.indexOf(clause,start);
    const firstMath=region.indexOf(variable+" * ",start);
    if(firstMath>=0) assert(guarded>=0 && guarded<firstMath);
  }
});
test("bootstrap accepted message magnitudes keep integer products inside signed 32-bit",()=>{
  for(const input of [-1000,0,1000]){
    const forward=input*3;
    assert(forward>=-3000&&forward<=3000);
    for(const upstream of [-4096,0,4096]){
      const vjp=input*upstream;
      assert(Number.isSafeInteger(vjp));
      assert(vjp>=-2147483648&&vjp<=2147483647);
      assert(Number.isSafeInteger(forward*upstream));
      assert(forward*upstream>=-2147483648&&forward*upstream<=2147483647);
    }
  }
});


test("input actor backward uses one version-bound saved-input snapshot",()=>{
  const first=source.slice(source.indexOf("define actor InputWeight"),
                           source.indexOf("define transform pipe actor HiddenWeight"));
  assert.match(first,/const \(saved_x, observed_version, was_issued, was_applied\)/);
  assert.match(first,/\(s\.saved_input, s\.version, s\.issued, s\.applied\)/);
  assert.match(first,/if !was_issued \|\| was_applied \|\| observed_version != 1/);
  assert.match(first,/s\.version != observed_version/);
  assert.equal((first.match(/using state\.read\(/g)||[]).length,1);
  assert(first.indexOf("const contribution = saved_x * dy") <
         first.indexOf("s.weight = s.weight - contribution"));
});
test("both trainable stages compare read-snapshot generation at publication",()=>{
  for(const region of [
    source.slice(source.indexOf("define actor InputWeight"),
                 source.indexOf("define transform pipe actor HiddenWeight")),
    source.slice(source.indexOf("define transform pipe actor HiddenWeight"),
                 source.indexOf("define actor SquaredLoss"))
  ]){
    assert.match(region,/using state\.read\(\|s\| ->/);
    assert.match(region,/if !s\.issued \|\| s\.applied \|\| s\.version != observed_version/);
  }
});
