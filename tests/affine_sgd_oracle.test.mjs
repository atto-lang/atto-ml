// HOST-ONLY reference, never an Oreslang compiler or runtime acceptance gate.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const source = readFileSync(new URL('../src/linear.ores', import.meta.url), 'utf8');
const guest = readFileSync(new URL('./smoke.ores', import.meta.url), 'utf8');
const finite = Number.isFinite;
const reject = (message) => { throw new Error(message); };

function sgd_into(parameters, gradients, rate) {
  if (parameters === gradients) reject('ALIAS');
  if (!parameters.length) reject('EMPTY');
  if (parameters.length !== gradients.length) reject('LENGTH');
  if (!(rate > 0) || !finite(rate)) reject('RATE');
  const next = parameters.map((p, i) => {
    const g = gradients[i];
    if (!finite(p) || !finite(g)) reject('NONFINITE_INPUT');
    const updated = p - rate * g;
    if (!finite(updated)) reject('NONFINITE_RESULT');
    return updated;
  });
  for (let i = 0; i < next.length; ++i) parameters[i] = next[i];
}
function affine_step(x, target, weights, biases, rate) {
  if (!x.length || x.length !== target.length) reject('BATCH');
  if (weights === biases || weights === x || weights === target || biases === x || biases === target) reject('ALIAS');
  if (weights.length !== 1 || biases.length !== 1) reject('PARAMETER_SHAPE');
  if (!(rate > 0) || !finite(rate)) reject('RATE');
  const w = weights[0], b = biases[0];
  if (![w, b, ...x, ...target].every(finite)) reject('NONFINITE_INPUT');
  let loss = 0, dw = 0, db = 0;
  for (let i = 0; i < x.length; i++) {
    const e = ((w * x[i]) + b) - target[i];
    loss += e * e;
    dw += e * x[i];
    db += e;
  }
  const nextW = w - (rate * (dw / x.length)) * 2;
  const nextB = b - (rate * (db / x.length)) * 2;
  if (![loss, nextW, nextB].every(finite)) reject('NONFINITE_RESULT');
  weights[0] = nextW;
  biases[0] = nextB;
  return loss / x.length;
}
const assert_atomic_failure = (call, arrays) => {
  const snapshots = arrays.map(a => a.slice());
  assert.throws(call);
  arrays.forEach((a,i) => assert.deepEqual(a,snapshots[i]));
};

test('guest source still uses checked two-phase SGD and affine publication', () => {
  assert.match(source, /pub fnc sgd_step_into\(/);
  assert.match(source, /if parameters is gradients/);
  assert.match(source, /sgd requires a nonempty parameter vector/);
  assert.match(source, /pub fnc train_affine_step\(/);
  assert.match(source, /weights\[0\] = next_weight;\s*biases\[0\] = next_bias;/);
  assert.match(guest, /ORES_ML_TRAINING_SMOKE_PASS/);
  assert.match(guest, /sgd_step_into\(rt mut borrow vector_params, rt borrow vector_gradient, 0\.5\)/);
});
test('multi-element SGD is componentwise and deterministic', () => {
  const a = [3, 5]; sgd_into(a, [2, 4], 0.5); assert.deepEqual(a, [2, 3]);
  sgd_into(a, [-2, 6], 0.5); assert.deepEqual(a, [3, 0]);
});
test('SGD fails atomically for alias, empty and length mismatch', () => {
  const v = [4,5]; assert_atomic_failure(() => sgd_into(v,v,0.5),[v]);
  const empty = []; assert_atomic_failure(() => sgd_into(empty,[],0.5),[empty]);
  assert_atomic_failure(() => sgd_into(v,[2],0.5),[v]);
});
test('SGD fails atomically for invalid rates and gradients', () => {
  const p=[7, 9];
  for (const rate of [0,-0.5,NaN,Infinity]) assert_atomic_failure(() => sgd_into(p,[1,1],rate),[p]);
  for (const bad of [NaN,Infinity,-Infinity]) {
    assert_atomic_failure(() => sgd_into(p,[1,bad],0.5),[p]);
    const invalidParameters = [1,bad];
    assert_atomic_failure(() => sgd_into(invalidParameters,[1,1],0.5),[invalidParameters]);
  }
});
test('late SGD overflow cannot partially publish first element', () => {
  const p = [8,Number.MAX_VALUE];
  assert_atomic_failure(() => sgd_into(p,[4,-Number.MAX_VALUE],2),[p]);
});
test('affine training converges on the guest fixture', () => {
  const x=[0,1,2,3], y=[1,3,5,7], w=[0], b=[0];
  let first=NaN, last=NaN;
  for(let k=0;k<500;k++) {const loss=affine_step(x,y,w,b,.05);if(k===0)first=loss;last=loss;}
  assert(first>last); assert(last<1e-6);
  assert(Math.abs(w[0]-2)<.01); assert(Math.abs(b[0]-1)<.01);
});
test('zero-loss affine step accepts a large finite rate', () => {
  const weights = [1], biases = [0];
  assert.equal(affine_step([2], [2], weights, biases, 1e308), 0);
  assert.deepEqual(weights, [1]);
  assert.deepEqual(biases, [0]);
});
test('affine training rejects malformed state without partial updates', () => {
  const x=[0,1], y=[1,3], w=[0], b=[0];
  assert_atomic_failure(() => affine_step(x,[1],w,b,.05),[w,b]);
  assert_atomic_failure(() => affine_step(x,y,w,w,.05),[w]);
  for(const rate of [0,-.05,NaN,Infinity])
    assert_atomic_failure(() => affine_step(x,y,w,b,rate),[w,b]);
  assert_atomic_failure(() => affine_step([],[],w,b,.05),[w,b]);
});
test('affine overflow after valid leading sample leaves both parameters unchanged', () => {
  const w=[2], b=[3];
  assert_atomic_failure(() => affine_step([0,Number.MAX_VALUE],[1,0],w,b,.05),[w,b]);
});
test('analytical affine gradients agree with independent central finite differences', () => {
  const x=[-.5,0,1.25,2.5], y=[1,1.5,3.5,6];
  const loss=(w,b)=>x.reduce((sum,xi,i)=>sum+((w*xi+b)-y[i])**2,0)/x.length;
  for (const [w,b] of [[0,0],[1.2,.8],[-.7,2.3]]) {
    const h=1e-5, scale=2/x.length;
    const dw=x.reduce((sum,xi,i)=>sum+(((w*xi+b)-y[i])*xi),0)*scale;
    const db=x.reduce((sum,xi,i)=>sum+((w*xi+b)-y[i]),0)*scale;
    assert(Math.abs(dw-(loss(w+h,b)-loss(w-h,b))/(2*h))<1e-8);
    assert(Math.abs(db-(loss(w,b+h)-loss(w,b-h))/(2*h))<1e-8);
  }
});
