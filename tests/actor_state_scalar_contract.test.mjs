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
  assert.doesNotMatch(source,/^\s*pub\s+(?:eval_affine|accumulate|commit|version)\(/m);
  assert.match(source,/^\s*private eval_affine\(/m);
  assert.doesNotMatch(source,/\b(?:private|pub)\s+(?:forward|backward|send|reply)\(/);
});

test("with effect bounds are independent from lexical using authority", () => {
  assert.match(source,/eval_affine\([^)]*\): Result<int, TrainingError>\s+with state\.read, state\.write/);
  assert.match(source,/using state\.read\(\|s\| -> s\.weight\)/);
  assert.match(source,/accumulate\([^)]*\): Result<int, TrainingError>\s+with state\.read, state\.write/);
  assert.match(source,/using state\.write\(critical \|s\| -> \{/);
  assert.match(source,/commit\([^)]*\): Result<int, TrainingError> with state\.read, state\.write/);
  assert.match(source,/self\.handle_state\(critical write \|s\| -> \{/);
  assert.match(source,/version\(\): int with state\.read/);
  assert.match(source,/s\.step != expected_step/);
  assert.match(source,/s\.seen_first = true/);
  assert.match(source,/s\.seen_second = true/);
  assert.match(source,/return Err\(TrainingError.DUPLICATE_GRADIENT\(\)\);/); // duplicate microbatch
  assert.match(source,/microbatch != 1 && microbatch != 2/);
  assert.match(source,/self\.handle_state\(critical read \|s\| -> s\.version\)/);
  assert.doesNotMatch(source,/\bfnc\s+\w+\([^)]*\)\s+using\s+state/);
});

test("write callback checks weight version and gradient count before publishing", () => {
  const start=source.indexOf("private commit("),end=source.indexOf("private version()",start);
  const commit=source.slice(start,end);
  const guard=commit.indexOf("s.version != expected_version");
  const full=commit.indexOf("s.pending_gradients != 2");
  const stepGuard=commit.indexOf("s.step != expected_step");
  const bothIds=commit.indexOf("!s.issued_first || !s.issued_second || !s.seen_first || !s.seen_second");
  assert(stepGuard > guard && bothIds > full);
  const update=commit.indexOf("s.weight = candidate_weight");
  const prepare=commit.indexOf("val int candidate_weight = s.weight - s.accumulated_gradient");
  const check=commit.indexOf("if candidate_weight < -10000 || candidate_weight > 10000");
  assert(prepare > full && check > prepare && update > check);
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
  // First microbatch repeated must not be counted twice; incomplete second
  // microbatch blocks a partial optimizer commit.
  const seen=new Set();
  const add=(id,g)=>{if(seen.has(id))return "duplicate";seen.add(id);return g;};
  assert.equal(add(1,contribution[0]),2);
  assert.equal(add(1,contribution[0]),"duplicate");
  assert.equal(seen.size,1);
  assert.equal(add(2,contribution[1]),3);
  assert.equal(seen.size,2);
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

test("expected guest-output golden includes orphan and duplicate-forward rejection", () => {
  const golden = readFileSync(new URL("./golden/actor_state_scalar.stdout", import.meta.url),"utf8");
  assert.deepEqual(golden.trim().split(/\r?\n/),
    ["REJECTED","REJECTED","20","REJECTED","30","REJECTED","2","REJECTED","REJECTED","5","5","2","REJECTED","REJECTED","REJECTED","REJECTED","50000","50000","100000000","200000000","REJECTED","2","3","REJECTED","10","-10","10"]);
  const messages=[...source.matchAll(/worker\.send\("([^"]+)"\)/g)].map(m=>m[1]);
  assert.deepEqual(messages,
    ["orphan","unsafe_forward","forward1","forward_duplicate","forward2","unsafe_upstream","grad1","duplicate","early","grad2","commit","stale","wrong_run","wrong_graph","wrong_incarnation","high_forward1","high_forward2","high_grad1","high_grad2","stale_after_cancel","new_forward","new_negative_forward"]);
});


test("every accumulate invocation has seven arguments and matching run/graph/incarnation",()=>{
  const calls=[...source.matchAll(/self\.accumulate\(([^)]*)\)/g)];
  assert.equal(calls.length,9);
  for(const call of calls) {
    const args=call[1].split(",").map(x=>x.trim());
    assert.equal(args.length,7);
    assert(["1","2"].includes(args[0]));
    assert.equal(args[0],args[1]);
    assert([1,2].includes(Number(args[2])));
    assert.equal(args[4],"101");
    assert.equal(args[5],"1");
    assert.equal(args[6],"202");
  }
  const stale=source.match(/if mail\.value == "stale"; do\s*self\.print_outcome\(self\.accumulate\(([^)]*)\)\)/);
  assert(stale);
  assert.equal(stale[1].split(",").map(x=>Number(x.trim()))[1],1);
  assert.match(source,/s\.version != expected_version/);
});
test("actor state effects and private helpers cannot be bypassed by handlers",()=>{
  const helpers=["eval_affine","accumulate","commit","version","weight_snapshot"];
  for(const name of helpers) {
    assert(source.includes("private "+name+"("));
    assert(!source.includes("pub "+name+"("));
  }
  const handler=source.slice(source.indexOf("receive(ActorMail<String> mail)"));
  assert.doesNotMatch(handler,/\bself\.state\b/);
  assert.doesNotMatch(handler,/\busing\s+state\.(?:read|write)\b/);
});

test("backprop consumes only recorded forward contexts and clears them on commit",()=>{
  assert.match(source,/issued_first: bool = false/);
  assert.match(source,/issued_second: bool = false/);
  assert.match(source,/input_first: int = 0/);
  assert.match(source,/input_second: int = 0/);
  assert.match(source,/if !s\.issued_first/);
  assert.match(source,/if !s\.issued_second/);
  assert.match(source,/s\.input_first \* upstream/);
  assert.match(source,/s\.input_second \* upstream/);
  assert.match(source,/s\.issued_first = false/);
  assert.match(source,/s\.issued_second = false/);
  assert.match(source,/s\.input_first = 0/);
  assert.match(source,/s\.input_second = 0/);
  assert.match(source,/return Err\(TrainingError.ORPHAN_GRADIENT\(\)\);/); // orphan backward
  assert.match(source,/return Err\(TrainingError.DUPLICATE_FORWARD\(\)\);/); // duplicate forward
  assert.doesNotMatch(source,/private accumulate\([^)]*\bint x\b/);
});


test("actor startup/shutdown uses canonical lifecycle methods, not property aliases",()=>{
  assert.match(source,/await worker\.get_ready\(\);/);
  assert.match(source,/await worker\.get_done\(\);/);
  assert.doesNotMatch(source,/await worker\.(?:ready|done)\s*;/);
  const ready=source.indexOf("await worker.get_ready();");
  const firstSend=source.indexOf('worker.send("orphan")');
  const lastSend=source.indexOf('worker.send("new_negative_forward")');
  const done=source.indexOf("await worker.get_done();");
  assert(ready>0&&firstSend>ready&&lastSend>firstSend&&done>lastSend);
});

test("bounded domain guards happen before receipt mutations",()=>{
  const forward=source.slice(source.indexOf("private eval_affine("),source.indexOf("private accumulate("));
  const backward=source.slice(source.indexOf("private accumulate("),source.indexOf("private commit("));
  const commit=source.slice(source.indexOf("private commit("),source.indexOf("private weight_snapshot("));
  const fguard=forward.indexOf("if x < -10000");
  const bguard=backward.indexOf("if upstream < -10000");
  const cguard=commit.indexOf("if candidate_weight < -10000");
  assert(fguard>0&&fguard<forward.indexOf("s.issued_first = true"));
  assert(bguard>0&&bguard<backward.indexOf("s.seen_first = true"));
  assert(cguard>0&&cguard<commit.indexOf("s.weight = candidate_weight"));
  assert.match(commit,/if s\.step >= 1000000 \|\| s\.version >= 1000000/);
  assert(10000*10000*2+10000<=2147483647);
});
test("bounded numeric rejects preserve source-gradient receipts and committed version",()=>{
  const stage={weight:5,version:2,step:2,input:10000,seen:false,gradient:0};
  const valid=x=>Number.isSafeInteger(x)&&x>=-10000&&x<=10000;
  const unsafe=10001;
  if(valid(unsafe))stage.seen=true;
  assert.equal(stage.seen,false);
  const contrib=stage.input*10000;
  stage.gradient+=contrib;stage.seen=true;
  stage.gradient+=contrib;
  assert.equal(stage.gradient,200000000);
  const candidate=stage.weight-stage.gradient;
  if(valid(candidate)){stage.weight=candidate;stage.version++;stage.step++;}
  assert.equal(stage.weight,5);
  assert.equal(stage.version,2);
  assert.equal(stage.step,2);
  const golden=readFileSync(new URL("./golden/actor_state_scalar.stdout",import.meta.url),"utf8")
     .trim().split(/\r?\n/);
  assert.deepEqual(golden.slice(-11),["50000","50000","100000000","200000000","REJECTED","2","3","REJECTED","10","-10","10"]);
});

test("typed cancellation handles only a fully received rejected candidate",()=>{
  const start=source.indexOf("private cancel_rejected(");
  const stop=source.indexOf("private weight_snapshot(",start);
  const body=source.slice(start,stop);
  assert.match(body,/Result<int, TrainingError>/);
  assert.match(body,/return Err\(TrainingError.STALE_STEP\(\)\)/);
  assert.match(body,/return Err\(TrainingError.INCOMPLETE_FANIN\(\)\)/);
  assert.match(body,/return Err\(TrainingError.COUNTER_EXHAUSTED\(\)\)/);
  assert.match(body,/return Err\(TrainingError.VALID_CANDIDATE_MUST_COMMIT\(\)\)/);
  assert.match(body,/return Ok\(s\.version\)/);
  const firstWrite=body.indexOf("s.accumulated_gradient = 0;");
  const guard=body.indexOf("if candidate >= -10000 && candidate <= 10000;");
  const versionBump=body.indexOf("s.version = s.version + 1;");
  assert(start>0&&guard>0&&firstWrite>guard&&versionBump>firstWrite);
  assert.doesNotMatch(body,/\bawait\b|\bselect\b|rt cooperate/);
});
test("rejected step retirement rejects old version and permits fresh forward",()=>{
  const stage={weight:5,version:2,step:2,gradient:200000000,issued:true,seen:true};
  const candidate=stage.weight-stage.gradient;
  assert(candidate < -10000);
  const committedBefore=stage.weight;
  stage.gradient=0;stage.issued=false;stage.seen=false;
  stage.version++;stage.step++;
  assert.equal(committedBefore,stage.weight);
  assert.equal(stage.version,3);
  assert.equal(stage.step,3);
  const oldVersion=2;
  assert(oldVersion!==stage.version);
  const firstFresh=stage.weight*2;
  assert.equal(firstFresh,10);
});

test("all actor training operations report Result errors without numeric sentinels",()=>{
  const start=source.indexOf("private eval_affine(");
  const stop=source.indexOf("// A controlled recovery path",start);
  const segment=source.slice(start,stop);
  for(const method of ["eval_affine","accumulate","commit"])
    assert.match(segment,new RegExp("private "+method+"\\([^)]*\\): Result<int, TrainingError>"));
  assert.doesNotMatch(segment,/return -[0-9]+;/);
  assert.match(segment,/return Err\(TrainingError.DUPLICATE_GRADIENT\(\)\)/);
  assert.match(segment,/return Err\(TrainingError.STALE_VERSION\(\)\)/);
  assert.match(segment,/return Ok\(s\.weight \* x\)/);
  assert.match(segment,/return Ok\(s\.accumulated_gradient\)/);
  assert.match(segment,/return Ok\(s\.weight\)/);
  assert.match(source,/private print_outcome\(Result<int, TrainingError> outcome\): void/);
  assert.match(source,/match outcome over/);
  assert.match(source,/match problem over/);
  assert.match(source,/stdio\.println\(value\)/);
});
test("a legitimate negative forward result cannot be confused with rejection",()=>{
  const output=readFileSync(new URL("./golden/actor_state_scalar.stdout",import.meta.url),"utf8")
    .trim().split(/\r?\n/);
  assert.equal(output[25],"-10");
  assert.equal(output[23],"REJECTED");
  assert.match(source,/self\.eval_affine\(3, 3, 2, -2, 101, 1, 202\)/);
  assert.equal(5*(-2),-10);
});


test("run and incarnation guards precede all scalar mutation and reject wrong epochs",()=>{
  assert.match(source,/run_id: int = 101/);
  assert.match(source,/incarnation: int = 1/);
  assert.match(source,/graph_id: int = 202/);
  for(const method of ["eval_affine","accumulate","commit","cancel_rejected"]){
    const start=source.indexOf("private "+method+"(");
    const end=source.indexOf("\n    private ",start+8);
    const body=source.slice(start,end<0?source.length:end);
    assert.match(body,/int expected_run, int expected_incarnation, int expected_graph/);
    const run=body.indexOf("s.run_id != expected_run");
    const graph=body.indexOf("s.graph_id != expected_graph");
    const inc=body.indexOf("s.incarnation != expected_incarnation");
    const version=body.indexOf("s.version != expected_version");
    const mutation=body.search(/s\.(?:issued_first|issued_second|seen_first|seen_second|weight|accumulated_gradient|step|version) =/);
    assert(run>0&&graph>run&&inc>graph&&version>inc,method+" identity fences before version");
    assert(mutation<0||mutation>version,method+" mutates only after fences");
    assert.match(body,/return Err\(TrainingError.STALE_RUN\(\)\)/);
    assert.match(body,/return Err\(TrainingError.STALE_GRAPH\(\)\)/);
    assert.match(body,/return Err\(TrainingError.STALE_INCAR\(\)\)/);
  }
  assert.match(source,/self\.eval_affine\(2, 2, 1, 2, 100, 1, 202\)/);
  assert.match(source,/self\.eval_affine\(2, 2, 1, 2, 101, 0, 202\)/);
  assert.match(source,/self\.eval_affine\(2, 2, 1, 2, 101, 1, 201\)/);
});
test("cancellation is not directly addressable as a string mail command",()=>{
  const handler=source.slice(source.indexOf("receive(ActorMail<String> mail)"));
  assert.doesNotMatch(handler,/mail\.value == "cancel_failed"/);
  assert.doesNotMatch(source,/worker\.send\("cancel_failed"\)/);
  assert.doesNotMatch(handler,/mail\.value == "high_commit"/);
  assert.doesNotMatch(source,/worker\.send\("high_commit"\)/);
  assert.doesNotMatch(handler,/mail\.value == "after"/);
  assert.doesNotMatch(source,/worker\.send\("after"\)/);
  assert.match(source,/private demo_finish_ready\(\): bool with state\.read/);
  assert.match(source,/s\.issued_first && s\.issued_second/);
  assert.match(handler,/!final_result\.is_err\(\) && self\.demo_finish_ready\(\)/);
  const decision=handler.slice(handler.indexOf('mail.value == "high_grad2"'),handler.indexOf('mail.value == "stale_after_cancel"'));
  assert.match(decision,/if !final_gradient\.is_err\(\)/);
  assert.match(decision,/self\.commit\(2, 2, 101, 1, 202\)/);
  assert.match(decision,/outcome\.is_err\(\)/);
  assert.match(decision,/self\.cancel_rejected\(2, 2, 101, 1, 202\)/);
  // This is an internal recovery decision, NOT caller authentication of other commands.
});

test("termination requires successful last-step forward and completed snapshot",()=>{
  const handler=source.slice(source.indexOf("receive(ActorMail<String> mail)"));
  const terminal=handler.slice(handler.indexOf('mail.value == "new_negative_forward"'),
      handler.indexOf("\n        return;\n    }\nend"));
  const result=terminal.indexOf("self.eval_affine(3, 3, 2, -2, 101, 1, 202)");
  const guard=terminal.indexOf("!final_result.is_err() && self.demo_finish_ready()");
  const end=terminal.indexOf("self.end()");
  assert(result>=0&&guard>result&&end>guard);
  assert.doesNotMatch(handler,/mail\.value == "after"/);
  const predicate=source.slice(source.indexOf("private demo_finish_ready("),
      source.indexOf("\n    receive("));
  for(const check of ["s.step == 3", "s.version == 3","s.run_id == 101","s.graph_id == 202",
      "s.incarnation == 1", "s.issued_first && s.issued_second", "s.pending_gradients == 0"]){
    assert(predicate.includes(check),check);
  }
});
