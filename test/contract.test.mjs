import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const c=JSON.parse(readFileSync(new URL("../integration/contract.v1.json",import.meta.url),"utf8"));
test("contract identifies exactly one source organization/repository",()=>{
 assert.equal(c.schemaVersion,1);assert.equal(c.organization,"ores-truffle-oreslang");
 const pkg=JSON.parse(readFileSync(new URL("../package.json",import.meta.url),"utf8"));
 assert.equal(c.repository,pkg.name.replace(/^@oreslang\//,""));
 assert.match(c.mainSnapshotSha,/^[a-f0-9]{40}$/);
});
test("published interface IDs and upstream edges are unique and canonical",()=>{
 for(const property of ["provides","requires","upstreams","fixtureIds"]){
  assert.ok(Array.isArray(c[property]));
  assert.deepEqual(c[property],[...new Set(c[property])].sort());
 }
 assert.ok(c.provides.length>0);
 assert.ok(c.fixtureIds.length>0);
 assert.ok(!c.upstreams.includes(c.repository));
 for(const req of c.requires) assert.ok(!c.provides.includes(req));
});
test("no host-oracle pass is elevated to Oreslang runtime proof",()=>{
 assert.equal(c.kind,"PROPOSED_CONTRACT_NOT_IMPLEMENTATION");
 assert.equal(c.hostOracleVerified,false);
 assert.equal(c.guestExecution,"NOT_EXECUTED");
 assert.deepEqual(c.evidence,[]);
});
test("next acceptance step must require executable semantics",()=>{
 assert.ok(c.nextGate.length>60);
 assert.match(c.nextGate,/(implement|compile|execute)/i);
});

test("v1 contract identity and reference provenance cannot silently drift",()=>{
 assert.deepEqual(Object.keys(c).sort(),[
  "schemaVersion","organization","repository","mainSnapshotSha","kind",
  "provides","requires","upstreams","fixtureIds","hostOracleRepo",
  "hostOracleVerified","guestExecution","evidence","nextGate"
 ].sort());
 // A historical bootstrap snapshot, not a claim that current HEAD was tested.
 assert.equal(c.mainSnapshotSha,"77893de4fb1f5a6b92e189f2a5f51ae520b22723");
 assert.deepEqual(c.provides,["oreslang.ml.linear-sgd.v1"]);
 assert.deepEqual(c.requires,["oreslang.autograd.scalar.v1","oreslang.tensor.cpu.v1"]);
 assert.deepEqual(c.upstreams,[
  "oreslang-autograd","oreslang-common-source","oreslang-tensor"
 ]);
 assert.deepEqual(c.fixtureIds,["ml/linear-sgd/v1"]);
 assert.equal(c.hostOracleRepo,"oreslang-ml-conformance");
});
