// Independent host oracle, not AttoLang actor or runtime execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

function neuron(weight) {
  let state = {weight, version: 1, step: 0, pending: null};
  return {
    snapshot: () => structuredClone(state),
    forward(step, version, x) {
      if (version !== state.version || step !== state.step + 1 || state.pending !== null ||
          !Number.isFinite(x) || !Number.isFinite(x * state.weight)) return null;
      state.pending = {step, x};
      return x * state.weight;
    },
    backward(step, version, upstream) {
      if (version !== state.version || state.pending?.step !== step || !Number.isFinite(upstream)) return null;
      const dx = upstream * state.weight;
      const dw = upstream * state.pending.x;
      const candidate = state.weight - .1 * dw;
      if (![dx, dw, candidate].every(Number.isFinite)) return null;
      state = {weight: candidate, version: state.version + 1, step, pending: null};
      return dx;
    }
  };
}
function unchanged(n, attempt) {
  const before = n.snapshot();
  assert.equal(attempt(), null);
  assert.deepEqual(n.snapshot(), before);
}
test('two persistent neurons propagate the old-weight derivative and improve the loss', () => {
  const h = neuron(2), o = neuron(1);
  const first = o.forward(1, 1, h.forward(1, 1, 1));
  assert.equal(first, 2);
  const dx = o.backward(1, 1, 2*(first-4));
  assert.equal(dx, -4);
  assert.equal(h.backward(1, 1, dx), -8);
  const second = o.forward(2, 2, h.forward(2, 2, 1));
  assert.ok(Math.abs(second-4.32) < 1e-12);
  assert.ok((second-4)**2 < (first-4)**2);
});
test('stale version, duplicate forward/backward and invalid step never publish', () => {
  const n = neuron(2);
  unchanged(n, () => n.backward(1, 1, 1));
  unchanged(n, () => n.forward(2, 1, 1));
  assert.equal(n.forward(1, 1, 1), 2);
  unchanged(n, () => n.forward(1, 1, 99));
  unchanged(n, () => n.backward(1, 0, 1));
  unchanged(n, () => n.backward(2, 1, 1));
  assert.equal(n.backward(1, 1, 1), 2);
  unchanged(n, () => n.backward(1, 1, 1));
});
test('nonfinite input and overflow leave state unchanged', () => {
  const n = neuron(2);
  unchanged(n, () => n.forward(1, 1, NaN));
  unchanged(n, () => n.forward(1, 1, Number.MAX_VALUE));
  assert.equal(n.forward(1, 1, 1), 2);
  unchanged(n, () => n.backward(1, 1, Infinity));
  unchanged(n, () => n.backward(1, 1, Number.MAX_VALUE));
  assert.equal(n.backward(1, 1, 1), 2);
});
test('guest fixture includes duplicate/stale controls and disclaims production authorization', () => {
  const src = readFileSync(new URL('../examples/train_stateful_neuron_microprocs.ores', import.meta.url), 'utf8');
  assert.match(src, /DUPLICATE_FORWARD/);
  assert.match(src, /STALE_VERSION/);
  assert.match(src, /trusted supervisor CONTROL/);
  assert.match(src, /ORES_ML_STATEFUL_NEURON_BACKPROP_PASS/);
  assert.match(src, /Not compiled at an exact compiler SHA/);
});
