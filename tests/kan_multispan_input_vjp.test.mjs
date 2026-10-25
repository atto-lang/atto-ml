// HOST-ONLY independent numerical oracle; not executable Oreslang code.
// Explicitly exercises the newly proposed checked input-VJP operation.
import test from 'node:test';
import assert from 'node:assert/strict';

const LIMIT = 1_000_000;
function basis(knots, i, degree, x) {
  if (degree === 0) return knots[i] <= x && x < knots[i+1] ? 1 : 0;
  const a = knots[i+degree]-knots[i];
  const b = knots[i+degree+1]-knots[i+1];
  return (a > 0 ? (x-knots[i])/a*basis(knots,i,degree-1,x) : 0)
    + (b > 0 ? (knots[i+degree+1]-x)/b*basis(knots,i+1,degree-1,x) : 0);
}
function evaluate(knots, coeff, x) {
  if (x === knots[coeff.length]) return coeff.at(-1);
  return coeff.reduce((total,c,i)=>total+c*basis(knots,i,3,x),0);
}
function finiteDerivative(knots, coeff, x) {
  // Analytic derivative from differentiated *basis recursion*, independent of
  // the implementation's degree-2 control/de Boor reduction.
  const n=coeff.length;
  if(x<knots[3]||x>knots[n])throw RangeError('DOMAIN');
  const atRight=x===knots[n];
  const dx=atRight?knots[n]-(knots[n]-knots[n-1])*1e-8:x;
  return coeff.reduce((total,c,i)=>{
    const a=knots[i+3]-knots[i], b=knots[i+4]-knots[i+1];
    return total+c*( (a>0?3/a*basis(knots,i,2,dx):0)
                     -(b>0?3/b*basis(knots,i+1,2,dx):0));
  },0);
}
function checkedInputVjp(knots,coeff,x,upstream) {
  if(!Number.isFinite(upstream)||Math.abs(upstream)>LIMIT)throw RangeError('UPSTREAM');
  const derivative=finiteDerivative(knots,coeff,x);
  if(!Number.isFinite(derivative)||Math.abs(derivative)>LIMIT)throw RangeError('DERIVATIVE');
  const result=derivative*upstream;
  if(!Number.isFinite(result)||Math.abs(result)>LIMIT)throw RangeError('PRODUCT');
  return result;
}
const near=(a,b,tolerance=1e-7)=>assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);
const knots=[0,0,0,0,.5,1,1,1,1];
const coeff=[0,.25,.5,.75,1];
let state=0x13242026;
const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/(2**32)};

test('checked input VJP matches central differences for 2,000 interior points',()=>{
  const step=1e-6;
  for(let i=0;i<2000;i++) {
    const x=.01+.98*random(), upstream=2*random()-1;
    const numeric=(evaluate(knots,coeff,x+step)-evaluate(knots,coeff,x-step))/(2*step)*upstream;
    near(checkedInputVjp(knots,coeff,x,upstream),numeric,3e-9);
  }
});
test('valid upstream produces 1.5 and zero adjoint produces zero',()=>{
  near(checkedInputVjp(knots,coeff,.5,2),1.5,1e-12);
  near(checkedInputVjp(knots,coeff,.5,0),0,1e-12);
  near(checkedInputVjp(knots,coeff,.5,-2),-1.5,1e-12);
});
test('bounded operands may produce a rejected VJP product',()=>{
  const steep=[0,1,2,3,4];
  near(checkedInputVjp(knots,steep,.5,1),3,1e-12);
  assert.throws(()=>checkedInputVjp(knots,steep,.5,1e6),/PRODUCT/);
  assert.throws(()=>checkedInputVjp(knots,steep,.5,-1e6),/PRODUCT/);
});
test('invalid upstream rejected before multiplication',()=>{
  for(const bad of [NaN,Infinity,-Infinity,LIMIT+1,-LIMIT-1])
    assert.throws(()=>checkedInputVjp(knots,coeff,.5,bad),/UPSTREAM/);
});
test('triple interior knot returns right-hand gradient on explicit branch',()=>{
  const ks=[0,0,0,0,.5,.5,.5,1,1,1,1];
  const c=[0,.1,.2,.3,.8,.5,.6];
  near(checkedInputVjp(ks,c,.5,2),6,1e-12);
  const h=1e-7;
  near((evaluate(ks,c,.5+h)-evaluate(ks,c,.5))/h,3,2e-5);
  near((evaluate(ks,c,.5)-evaluate(ks,c,.5-h))/h,.6,2e-5);
});
