// HOST-ONLY independent reference of typed actor-state transactions.
// It inspects the source contract but DOES NOT run the Oreslang compiler.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../examples/train_actor_state_scalar.ores",import.meta.url),"utf8");
const within=n=>Number.isSafeInteger(n)&&Math.abs(n)<=10000;
const ok=value=>Object.freeze({ok:true,value});
const err=reason=>Object.freeze({ok:false,reason});
class Stage {
  constructor(weight=10,step=1,version=1,run=101,inc=1){
    this.weight=weight;this.step=step;this.version=version;this.run=run;this.incarnation=inc;
    this.forward=new Map();this.gradients=new Map();
  }
  forwardStep(step,version,id,x,run=this.run,inc=this.incarnation){
    if(run!==this.run)return err("STALE_RUN");
    if(inc!==this.incarnation)return err("STALE_INCAR");
    if(version!==this.version)return err("STALE_VERSION");
    if(step!==this.step)return err("STALE_STEP");
    if(!within(x)||!within(this.weight))return err("INVALID_INPUT");
    if(id!==1&&id!==2)return err("INVALID_MICROBATCH");
    if(this.forward.has(id))return err("DUPLICATE_FORWARD");
    this.forward.set(id,x);return ok(this.weight*x);
  }
  backwardStep(step,version,id,u,run=this.run,inc=this.incarnation){
    if(run!==this.run)return err("STALE_RUN");
    if(inc!==this.incarnation)return err("STALE_INCAR");
    if(version!==this.version)return err("STALE_VERSION");
    if(step!==this.step)return err("STALE_STEP");
    if(id!==1&&id!==2)return err("INVALID_MICROBATCH");
    if(!within(u))return err("INVALID_GRADIENT");
    if(!this.forward.has(id))return err("ORPHAN_GRADIENT");
    if(this.gradients.has(id))return err("DUPLICATE_GRADIENT");
    const contribution=this.forward.get(id)*u;
    this.gradients.set(id,contribution);
    return ok([...this.gradients.values()].reduce((s,v)=>s+v,0));
  }
  candidate(){
    if(this.forward.size!==2||this.gradients.size!==2)return err("INCOMPLETE_FANIN");
    if(this.step>=1000000||this.version>=1000000)return err("COUNTER_EXHAUSTED");
    const gradient=[...this.gradients.values()].reduce((s,v)=>s+v,0);
    const value=this.weight-gradient;
    return within(value)?ok(value):err("UNSAFE_CANDIDATE");
  }
  commit(step,version,run=this.run,inc=this.incarnation){
    if(run!==this.run)return err("STALE_RUN");
    if(inc!==this.incarnation)return err("STALE_INCAR");
    if(version!==this.version)return err("STALE_VERSION");
    if(step!==this.step)return err("STALE_STEP");
    const p=this.candidate();if(!p.ok)return p;
    this.weight=p.value;this.step++;this.version++;this.forward.clear();this.gradients.clear();
    return ok(this.weight);
  }
  cancelRejected(step,version,run=this.run,inc=this.incarnation){
    if(run!==this.run)return err("STALE_RUN");
    if(inc!==this.incarnation)return err("STALE_INCAR");
    if(version!==this.version||step!==this.step)return err("STALE_STEP");
    const p=this.candidate();
    if(p.ok)return err("VALID_CANDIDATE_MUST_COMMIT");
    if(p.reason!=="UNSAFE_CANDIDATE")return p;
    this.step++;this.version++;this.forward.clear();this.gradients.clear();
    return ok(this.version);
  }
  snapshot(){return {weight:this.weight,step:this.step,version:this.version,
    forwards:this.forward.size,receipts:this.gradients.size};}
}
test("all three stateful operations and compensator return disjoint typed outcomes",()=>{
  for(const op of ["eval_affine","accumulate","commit","cancel_rejected"])
    assert.match(source,new RegExp("private "+op+"\\([^)]*\\): Result<int, String>"));
  assert.match(source,/if outcome\.is_err\(\); do/);
  assert.doesNotMatch(source.slice(source.indexOf("private eval_affine("),
    source.indexOf("// A controlled recovery path")),/return -\d+;/);
});
test("valid negative prediction is not mistaken for an error",()=>{
  const stage=new Stage(-2);
  assert.deepEqual(stage.forwardStep(1,1,1,4),ok(-8));
  assert.deepEqual(stage.forwardStep(1,1,2,10001),err("INVALID_INPUT"));
});
test("invalid messages and duplicate receipts do not mutate the snapshot",()=>{
  const stage=new Stage(),initial=stage.snapshot();
  assert.deepEqual(stage.backwardStep(1,1,1,1),err("ORPHAN_GRADIENT"));
  assert.deepEqual(stage.forwardStep(1,1,1,10001),err("INVALID_INPUT"));
  assert.deepEqual(stage.snapshot(),initial);
  assert.deepEqual(stage.forwardStep(1,1,1,2),ok(20));
  const now=stage.snapshot();
  assert.deepEqual(stage.forwardStep(1,1,1,2),err("DUPLICATE_FORWARD"));
  assert.deepEqual(stage.backwardStep(1,1,1,10001),err("INVALID_GRADIENT"));
  assert.deepEqual(stage.snapshot(),now);
  assert.deepEqual(stage.backwardStep(1,1,1,1),ok(2));
  const once=stage.snapshot();
  assert.deepEqual(stage.backwardStep(1,1,1,1),err("DUPLICATE_GRADIENT"));
  assert.deepEqual(stage.snapshot(),once);
});
test("incomplete fan-in blocks optimizer and unauthorized cancellation",()=>{
  const s=new Stage();s.forwardStep(1,1,1,2);s.backwardStep(1,1,1,1);
  const before=s.snapshot();
  assert.deepEqual(s.commit(1,1),err("INCOMPLETE_FANIN"));
  assert.deepEqual(s.cancelRejected(1,1),err("INCOMPLETE_FANIN"));
  assert.deepEqual(s.snapshot(),before);
});
test("complete valid candidate commits atomically and fences stale messages",()=>{
  const s=new Stage();
  for(const [id,x] of [[1,2],[2,3]])s.forwardStep(1,1,id,x);
  s.backwardStep(1,1,1,1);s.backwardStep(1,1,2,1);
  assert.deepEqual(s.cancelRejected(1,1),err("VALID_CANDIDATE_MUST_COMMIT"));
  assert.deepEqual(s.commit(1,1),ok(5));
  assert.deepEqual(s.snapshot(),{weight:5,step:2,version:2,forwards:0,receipts:0});
  assert.deepEqual(s.backwardStep(1,1,1,1),err("STALE_VERSION"));
});
test("failed complete candidate preserves committed state until compensation",()=>{
  const s=new Stage(5,2,2);
  for(const id of [1,2])s.forwardStep(2,2,id,10000);
  s.backwardStep(2,2,1,10000);s.backwardStep(2,2,2,10000);
  const before=s.snapshot();
  assert.deepEqual(s.commit(2,2),err("UNSAFE_CANDIDATE"));
  assert.deepEqual(s.snapshot(),before);
  assert.deepEqual(s.cancelRejected(2,2),ok(3));
  assert.deepEqual(s.snapshot(),{weight:5,step:3,version:3,forwards:0,receipts:0});
  assert.deepEqual(s.backwardStep(2,2,1,1),err("STALE_VERSION"));
  assert.deepEqual(s.forwardStep(3,3,2,-2),ok(-10));
});
test("exhausted counters never allow commit or compensate",()=>{
  for(const [s,v] of [[1000000,2],[2,1000000]]){
    const actor=new Stage(5,s,v);
    for(const id of [1,2])actor.forwardStep(s,v,id,10000);
    for(const id of [1,2])actor.backwardStep(s,v,id,10000);
    const before=actor.snapshot();
    assert.deepEqual(actor.commit(s,v),err("COUNTER_EXHAUSTED"));
    assert.deepEqual(actor.cancelRejected(s,v),err("COUNTER_EXHAUSTED"));
    assert.deepEqual(actor.snapshot(),before);
  }
});
test("4096 seeded bounded two-message schedules are fail-atomic",()=>{
  let seed=20261018;const next=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
  for(let i=0;i<4096;i++){
    const weight=(next()%20001)-10000,x1=(next()%20001)-10000;
    const x2=(next()%20001)-10000,u1=(next()%20001)-10000;
    const u2=(next()%20001)-10000;
    const s=new Stage(weight);
    s.forwardStep(1,1,1,x1);s.forwardStep(1,1,2,x2);
    s.backwardStep(1,1,1,u1);s.backwardStep(1,1,2,u2);
    const before=s.snapshot(),candidate=weight-x1*u1-x2*u2;
    const actual=s.commit(1,1);
    if(within(candidate)){
      assert.deepEqual(actual,ok(candidate));
      assert.equal(s.version,2);
    }else{
      assert.deepEqual(actual,err("UNSAFE_CANDIDATE"));
      assert.deepEqual(s.snapshot(),before);
    }
  }
});

test("stale runs and reused actor incarnations cannot reserve or publish",()=>{
  const stage=new Stage(5,2,2,101,3);
  const before=stage.snapshot();
  for(const run of [1,100,102,1000000]){
    assert.deepEqual(stage.forwardStep(2,2,1,4,run,3),err("STALE_RUN"));
    assert.deepEqual(stage.backwardStep(2,2,1,4,run,3),err("STALE_RUN"));
    assert.deepEqual(stage.commit(2,2,run,3),err("STALE_RUN"));
    assert.deepEqual(stage.cancelRejected(2,2,run,3),err("STALE_RUN"));
  }
  for(const inc of [0,1,2,4]){
    assert.deepEqual(stage.forwardStep(2,2,1,4,101,inc),err("STALE_INCAR"));
    assert.deepEqual(stage.backwardStep(2,2,1,4,101,inc),err("STALE_INCAR"));
    assert.deepEqual(stage.commit(2,2,101,inc),err("STALE_INCAR"));
    assert.deepEqual(stage.cancelRejected(2,2,101,inc),err("STALE_INCAR"));
  }
  assert.deepEqual(stage.snapshot(),before);
});
test("restart at an identical step/version still fences previous incarnation",()=>{
  const old=new Stage(5,1,1,101,1),restarted=new Stage(5,1,1,101,2);
  old.forwardStep(1,1,1,4);
  assert.deepEqual(restarted.forwardStep(1,1,1,4,101,1),err("STALE_INCAR"));
  assert.deepEqual(restarted.backwardStep(1,1,1,1,101,1),err("STALE_INCAR"));
  assert.deepEqual(restarted.forwardStep(1,1,1,4,101,2),ok(20));
});
test("8192 seeded replay/epoch requests preserve all state",()=>{
  const stage=new Stage(5,9,9,101,17);
  let seed=0xa17e2026;
  const rand=()=>seed=(Math.imul(seed,1664525)+1013904223)>>>0;
  for(let i=0;i<8192;i++){
    const run=rand()%2?100:102,inc=rand()%2?16:18;
    const kind=rand()%4,before=stage.snapshot();
    const actual=kind===0?stage.forwardStep(9,9,1,2,run,inc)
        :kind===1?stage.backwardStep(9,9,1,1,run,inc)
        :kind===2?stage.commit(9,9,run,inc)
        :stage.cancelRejected(9,9,run,inc);
    assert.deepEqual(actual,err("STALE_RUN"));
    assert.deepEqual(stage.snapshot(),before);
  }
});
