// Source-local contract regression only. NOT native guest, numerical or AOT execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const c = JSON.parse(readFileSync(new URL('../integration/contract.v1.json', import.meta.url), 'utf8'));

test('SGD may only advertise the exact tensor/autograd dependency chain', () => {
  assert.deepEqual(c.provides, ['oreslang.ml.linear-sgd.v1']);
  assert.deepEqual(c.requires, ['oreslang.autograd.scalar.v1', 'oreslang.tensor.cpu.v1']);
  assert.deepEqual(c.upstreams, ['oreslang-autograd','oreslang-common-source','oreslang-tensor']);
  assert.ok(c.requires.every(id => !c.provides.includes(id)));
  assert.ok(!c.upstreams.includes(c.repository));
  assert.deepEqual(c.fixtureIds, ['ml/linear-sgd/v1']);
});

test('independent host numeric oracle cannot certify runtime implementation', () => {
  assert.equal(c.kind, 'PROPOSED_CONTRACT_NOT_IMPLEMENTATION');
  assert.equal(c.hostOracleRepo, 'oreslang-ml-conformance');
  assert.equal(c.hostOracleVerified, false);
  assert.equal(c.guestExecution, 'NOT_EXECUTED');
  assert.deepEqual(c.evidence, []);
  assert.match(c.nextGate, /real Java\/LLVM guest execution/i);
  assert.match(c.nextGate, /decreasing loss/i);
});

test('bootstrap pin is historical evidence rather than current-head attestation', () => {
  assert.match(c.mainSnapshotSha, /^[0-9a-f]{40}$/);
  assert.equal(c.mainSnapshotSha, '77893de4fb1f5a6b92e189f2a5f51ae520b22723');
  assert.equal(c.repository, 'oreslang-ml');
  assert.equal(c.organization, 'ores-truffle-oreslang');
});
