import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const contract = JSON.parse(readFileSync(new URL('../integration/contract.v1.json', import.meta.url), 'utf8'));

test('ML proposed capability has bounded, namespaced and versioned identities', () => {
  for (const id of [...contract.provides, ...contract.requires]) {
    assert.match(id, /^oreslang\.[a-z][a-z0-9.-]*\.v[1-9][0-9]*$/);
    assert.ok(!id.includes('..') && !id.includes('/') && !id.includes('\\'));
  }
  assert.deepEqual(contract.provides, ['oreslang.ml.linear-sgd.v1']);
});

test('SGD contract cannot silently shed tensor and autograd prerequisites', () => {
  assert.deepEqual(contract.requires, [
    'oreslang.autograd.scalar.v1',
    'oreslang.tensor.cpu.v1',
  ]);
  assert.ok(contract.upstreams.includes('oreslang-autograd'));
  assert.ok(contract.upstreams.includes('oreslang-tensor'));
  assert.ok(contract.upstreams.includes('oreslang-common-source'));
});

test('host-only numeric reference never certifies compiled gradient execution', () => {
  assert.equal(contract.kind, 'PROPOSED_CONTRACT_NOT_IMPLEMENTATION');
  assert.equal(contract.hostOracleRepo, 'oreslang-ml-conformance');
  assert.equal(contract.hostOracleVerified, false);
  assert.equal(contract.guestExecution, 'NOT_EXECUTED');
  assert.deepEqual(contract.evidence, []);
  assert.match(contract.nextGate, /real Java\/LLVM guest execution/i);
  assert.deepEqual(contract.fixtureIds, ['ml/linear-sgd/v1']);
});
