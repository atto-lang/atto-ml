// HOST-ONLY numerical and source-contract tests.
// Do not treat these as execution of the .ores guest, actor runtime or tape.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const source=readFileSync(new URL("../examples/train_actor_two_layer_autograd.ores",import.meta.url),"utf8");
const pairs=[[0,1],[1,3],[2,5],[3,7]];
const params=Object.freeze({w1:1,b1:1,w2:1,b2:0});
function sample(p,x,y) {
  const z=p.w1*x+p.b1, h=Math.max(0,z);
  const d=p.w2*h+p.b2-y, scale=2*d, dz=z>0?scale*p.w2:0;
  return {
    loss:d*d,w1:dz*x,b1:dz,w2:scale*h,b2:scale,
    input:dz
  };
}
function batch(p) {
  const sum={w1:0,b1:0,w2:0,b2:0,loss:0};
  for(const [x,y] of pairs){
    const g=sample(p,x,y);
    for(const k of Object.keys(sum)) sum[k]+=g[k]/pairs.length;
  }
  return sum;
}
function finite(p) {return Object.values(p).every(Number.isFinite);}
function candidate(p,lr=0.02) {
  const g=batch(p), next={};
  for(const k of ["w1","b1","w2","b2"])next[k]=p[k]-lr*g[k];
  return {next,loss:g.loss};
}
// A separate host oracle for the state-transition CAS; production runtime
// must enforce exclusivity and ownership with its real sealed gate.
class Stage {
  constructor() {
    this.state={...params,run:101,graph:202,incarnation:1,version:1,committedEpoch:0};
  }
  read() {return Object.freeze({...this.state});}
  commit(snap, next, epoch, expectedRun=snap.run, expectedGraph=snap.graph,
         expectedIncarnation=snap.incarnation) {
    if(!finite(next))return "NONFINITE_CANDIDATE";
    const s=this.state;
    if(s.run!==snap.run || snap.run!==expectedRun)return "STALE_RUN";
    if(s.graph!==snap.graph || snap.graph!==expectedGraph)return "STALE_GRAPH";
    if(s.incarnation!==snap.incarnation || snap.incarnation!==expectedIncarnation)
      return "STALE_INCAR";
    if(s.version!==snap.version || s.committedEpoch!==snap.committedEpoch
       || epoch!==s.committedEpoch+1)return "STALE_SNAPSHOT";
    this.state={...next,run:s.run,graph:s.graph,incarnation:s.incarnation,
      version:s.version+1,committedEpoch:epoch};
    return "OK";
  }
}
test("actor has one state schema, only primitive snapshots and one all-parameter CAS",()=>{
  assert.equal((source.match(/\bstate\s*\{/g)||[]).length,1);
  assert.doesNotMatch(source,/self\.state\b/);
  assert.match(source,/using state\.read\(\|s\|\s*->\s*\(/);
  assert.match(source,/pinned_run, pinned_graph, pinned_incarnation/);
  assert.match(source,/Result<float, TwoLayerTrainingError>/);
  assert.doesNotMatch(source,/Result<float, String>|return Err\("/);
  for(const gate of ["run_id","graph_id","incarnation"]){
    assert.match(source,new RegExp(gate+": int ="));
  }
  assert.match(source,/self\.handle_state\(critical write \|s\| -> \{/);
  assert.match(source,/s\.run_id != pinned_run \|\| pinned_run != expected_run/);
  assert.match(source,/s\.graph_id != pinned_graph \|\| pinned_graph != expected_graph/);
  assert.match(source,/s\.incarnation != pinned_incarnation/);
  assert.match(source,/s\.version != pinned_version \|\| s\.committed_epoch != prior_epoch/);
  assert.match(source,/import class TapeF64/);
  assert.match(source,/forward_checked_into\(/);
  assert.match(source,/backward_checked_into\(/);
  const commit=source.slice(source.lastIndexOf("return self.handle_state(critical write"));
  for(const name of ["first_weight","first_bias","second_weight","second_bias","version","committed_epoch"]){
    assert.match(commit,new RegExp("s\\."+name+"\\s*="));
  }
  // The tape and SGD calculations precede the write transaction.
  assert(source.indexOf("backward_checked_into(")<source.indexOf("return self.handle_state(critical write"));
  assert(source.indexOf("candidate_b2 =")<source.indexOf("return self.handle_state(critical write"));
  assert.doesNotMatch(commit.slice(0,commit.indexOf("});")),/\bawait\b|rt cooperate|self\.forward\(/);
  assert.match(source,/await worker\.get_ready\(\)/);
  assert.match(source,/await worker\.get_done\(\)/);
});
test("source uses autograd rather than handwritten gradient formulas",()=>{
  assert.match(source,/val TapeF64 tape = new TapeF64\(/);
  assert.match(source,/gradients\[0\]/);
  assert.match(source,/gradients\[2\]/);
  assert.match(source,/gradients\[3\]/);
  assert.match(source,/gradients\[4\]/);
  assert.match(source,/TwoLayerTrainingError\.NONFINITE_GRADIENT\(\)/);
  assert.match(source,/TwoLayerTrainingError\.NONFINITE_CANDIDATE\(\)/);
  const declaration=source.match(/define enum TwoLayerTrainingError as\n([\s\S]*?)\nend/);
  assert(declaration);
  const cases=[...declaration[1].matchAll(/^\s+([A-Z_]+) = (\d+)$/gm)].map(m=>m[1]);
  assert.deepEqual(cases,["STALE_RUN","STALE_GRAPH","STALE_INCAR","STALE_EPOCH",
    "COUNTER_EXHAUSTED","FORWARD_REJECTED","BACKWARD_REJECTED",
    "NONFINITE_GRADIENT","NONFINITE_CANDIDATE","STALE_SNAPSHOT"]);
  const failureMatch=source.slice(source.indexOf("match problem over"),
    source.indexOf("end",source.indexOf("match problem over")));
  for(const name of cases)assert.match(failureMatch,new RegExp("on TwoLayerTrainingError\\."+name+" ->"));
  assert.doesNotMatch(failureMatch,/on _|default/);
});
test("independent per-parameter gradients pass finite differences",()=>{
  for(const p of [params,{w1:.7,b1:.5,w2:1.3,b2:-.2}]){
    const g=batch(p),eps=1e-6;
    for(const key of ["w1","b1","w2","b2"]){
      const hi={...p,[key]:p[key]+eps},lo={...p,[key]:p[key]-eps};
      const numerical=(batch(hi).loss-batch(lo).loss)/(2*eps);
      assert(Math.abs(numerical-g[key])<1e-6,key+" finite-difference mismatch");
    }
  }
});
test("ReLU negative branch blocks hidden gradients but not output bias",()=>{
  const g=sample({w1:1,b1:-20,w2:2,b2:0},3,7);
  assert.equal(g.w1,0);assert.equal(g.b1,0);assert(Math.abs(g.w2)===0);
  assert.equal(g.input,0);assert.equal(g.b2,-14);
});
test("1000 independently modeled SGD commits converge and publish once per epoch",()=>{
  const stage=new Stage();
  let lastLoss=Infinity;
  for(let epoch=1;epoch<=1000;epoch++){
    const pin=stage.read(),{next,loss}=candidate(pin);
    lastLoss=loss;
    assert.equal(stage.commit(pin,next,epoch),"OK");
  }
  assert(lastLoss<1e-6,"final loss did not converge");
  assert.equal(stage.state.version,1001);
  assert.equal(stage.state.committedEpoch,1000);
  for(const [x,y] of pairs){
    const p=stage.state,z=p.w1*x+p.b1;
    assert(Math.abs(p.w2*Math.max(0,z)+p.b2-y)<0.01);
  }
});
test("two computations from same pinned version cannot both publish",()=>{
  const stage=new Stage(),pin=stage.read(),{next}=candidate(pin);
  assert.equal(stage.commit(pin,next,1),"OK");
  const before=structuredClone(stage.state);
  assert.equal(stage.commit(pin,next,1),"STALE_SNAPSHOT");
  assert.deepEqual(stage.state,before);
  assert.equal(pin.version,1);
  assert.equal(pin.w1,1); // copied old version remains immutable
});
test("run, graph and incarnation mismatches are fail-atomic before snapshot publication",()=>{
  for(const [field,bad,reason] of [
    ["run",100,"STALE_RUN"],["graph",201,"STALE_GRAPH"],["incarnation",0,"STALE_INCAR"]
  ]){
    const stage=new Stage(),snap=stage.read(),{next}=candidate(snap);
    const before=structuredClone(stage.state);
    const args={run:snap.run,graph:snap.graph,incarnation:snap.incarnation,[field]:bad};
    assert.equal(stage.commit(snap,next,1,args.run,args.graph,args.incarnation),reason);
    assert.deepEqual(stage.state,before);
  }
  for(const [field,bad,reason] of [
    ["run",102,"STALE_RUN"],["graph",203,"STALE_GRAPH"],["incarnation",2,"STALE_INCAR"]
  ]){
    const stage=new Stage(),snap=stage.read(),{next}=candidate(snap);
    stage.state={...stage.state,[field]:bad};
    const before=structuredClone(stage.state);
    assert.equal(stage.commit(snap,next,1),reason);
    assert.deepEqual(stage.state,before);
  }
});

test("version agreement alone is insufficient: epoch and order also fence writes",()=>{
  const stage=new Stage(),snap=stage.read(),{next}=candidate(snap);
  assert.equal(stage.commit(snap,next,2),"STALE_SNAPSHOT");
  assert.equal(stage.commit({...snap,committedEpoch:-1},next,1),"STALE_SNAPSHOT");
  assert.deepEqual(stage.state,{...params,run:101,graph:202,incarnation:1,version:1,committedEpoch:0});
});
test("nonfinite candidate is rejected before any of four parameters change",()=>{
  for(const bad of [Infinity,-Infinity,NaN]){
    const stage=new Stage(),pin=stage.read();
    const next={...candidate(pin).next,w2:bad};
    assert.equal(stage.commit(pin,next,1),"NONFINITE_CANDIDATE");
    assert.deepEqual(stage.state,{...params,run:101,graph:202,incarnation:1,version:1,committedEpoch:0});
  }
});
test("stale-step interleaving deterministically rejects across parameter candidates",()=>{
  for(let i=0;i<128;i++){
    const stage=new Stage(),pin=stage.read(),{next}=candidate(pin);
    const altered={...next,b2:next.b2+(i+1)*.001};
    assert.equal(stage.commit(pin,i%2?altered:next,1),"OK");
    const expected=structuredClone(stage.state);
    assert.equal(stage.commit(pin,i%2?next:altered,1),"STALE_SNAPSHOT");
    assert.deepEqual(stage.state,expected);
  }
});
