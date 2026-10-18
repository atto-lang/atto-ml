// Host-only contract; never substitute for compiled Oreslang typechecking.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../examples/train_actor_state_scalar.ores",import.meta.url),"utf8");
const cases=["STALE_RUN","STALE_GRAPH","STALE_INCAR","STALE_VERSION","STALE_STEP","INVALID_INPUT","DUPLICATE_FORWARD","INVALID_MICROBATCH","INVALID_GRADIENT","ORPHAN_GRADIENT","DUPLICATE_GRADIENT","INCOMPLETE_FANIN","COUNTER_EXHAUSTED","UNSAFE_CANDIDATE","VALID_CANDIDATE_MUST_COMMIT"];
test("all actor training errors are closed nominal enum variants",()=>{
  const declaration=source.match(/define enum TrainingError as\\n([\\s\\S]*?)\\nend/);
  assert(declaration,"missing enum");
  const declared=[...declaration[1].matchAll(/^\\s+([A-Z_]+) = (\\d+)$/gm)];
  assert.deepEqual(declared.map(m=>m[1]),cases);
  assert.deepEqual(declared.map(m=>Number(m[2])),cases.map((_,i)=>i+1));
  assert.doesNotMatch(source,/Result<int, String>|Err\\("/);
});
test("all four methods use the nominal result and never emit freeform strings",()=>{
  for(const op of ["eval_affine","accumulate","commit","cancel_rejected"])
    assert.match(source,new RegExp("private "+op+"\\\\([^)]*\\\\): Result<int, TrainingError>"));
  const used=[...source.matchAll(/Err\\(TrainingError\\.([A-Z_]+)\\(\\)\\)/g)].map(m=>m[1]);
  assert(used.length>=20);
  assert(used.every(v=>cases.includes(v)));
});
test("closed error matcher handles every declared variant exactly once and has no catch-all",()=>{
  const match=source.slice(source.indexOf("match problem over"),source.indexOf("end",source.indexOf("match problem over")));
  const handled=[...match.matchAll(/on TrainingError\\.([A-Z_]+) ->/g)].map(m=>m[1]);
  assert.deepEqual(handled,cases);
  assert.doesNotMatch(match,/on _|on else|default/);
  assert.match(source,/on Ok\\(value\\)/);
  assert.match(source,/on Err\\(problem\\)/);
});
test("every persistent mutation remains after all run graph incarnation and epoch checks",()=>{
  for(const op of ["eval_affine","accumulate","commit","cancel_rejected"]){
    const start=source.indexOf("private "+op+"(");
    const end=source.indexOf("\\n    private ",start+8);
    const body=source.slice(start,end<0?source.length:end);
    const gates=["s.run_id != expected_run","s.graph_id != expected_graph","s.incarnation != expected_incarnation","s.version != expected_version"];
    for(const gate of gates)assert(body.includes(gate),op+" missing "+gate);
  }
});
