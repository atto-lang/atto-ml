import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {
  cubicBasis,cubicBasisDerivative,cubicEdge,cubicEdgeVjp,
  kanLayerForward,kanLayerVjp,mfkanForward,mfkanVjp
} from '../reference/kan_mfkan_reference.mjs';

const close=(a,b,eps=2e-7)=>assert.ok(Math.abs(a-b)<=eps,`${a} != ${b}`);
const derivative=(f,t,h=1e-6)=>(f(t+h)-f(t-h))/(2*h);
const layer=(inputs,outputs,coeff,bias)=>({inputs,outputs,coeff,bias});
function fixture() {
  return {
    low:layer(2,1,[.15,.24,.42,.54,.12,.18,.30,.48],[.05]),
    highNonlinear:layer(3,1,[.1,.2,.4,.5,.3,.4,.5,.6,.2,.3,.4,.5],[.1]),
    highLinear:[.3,.2,.5,-.1], mix:.4,
  };
}
const source=readFileSync(new URL('../src/kan/cubic_bspline_f64.ores',import.meta.url),'utf8');
const example=readFileSync(new URL('../examples/kan_mfkan_scalar_smoke.ores',import.meta.url),'utf8');

test('source defines Oreslang-native forward and VJP primitives (guest not executed)',()=>{
  for (const name of ['kan_cubic_segment_f64','kan_cubic_segment_dinput_f64','kan_cubic_segment_dcoeff_f64','mfkan_residual_blend_f64'])
    assert.match(source,new RegExp(`pub fnc ${name}\\(`));
  assert.match(example,/ORES_KAN_MFKAN_SCALAR_PASS/);
  assert.doesNotMatch(source,/\b(?:python|ffi|native_call)\b/i);
});
test('uniform cubic basis has partition of unity and known midpoint',()=>{
  for(let j=0;j<=100;j++) {
    const b=cubicBasis(j/100), d=cubicBasisDerivative(j/100);
    close(b.reduce((s,v)=>s+v,0),1,1e-14);
    close(d.reduce((s,v)=>s+v,0),0,1e-14);
    assert(b.every(v=>v>=0));
  }
  const m=cubicBasis(.5);
  for(let i=0;i<4;i++)close(m[i],[1,23,23,1][i]/48,1e-15);
});
test('constant coefficients preserve constant output for all local t',()=>{
  for(let i=0;i<=40;i++)close(cubicEdge(i/40,[.7,.7,.7,.7]),.7,1e-14);
});
test('analytic basis derivatives agree with central finite differences',()=>{
  for(const t of [.03,.1,.23,.51,.79,.93])for(let k=0;k<4;k++)
    close(cubicBasisDerivative(t)[k],derivative(x=>cubicBasis(x)[k],t),1e-9);
});
test('edge VJP agrees with independent finite differences for all coefficients and input',()=>{
  const cs=[-.3,.4,.7,1.2];
  for(const t of [.12,.36,.78]) {
    const g=cubicEdgeVjp(t,cs,-.9);
    close(g.inputGrad,-.9*derivative(x=>cubicEdge(x,cs),t));
    for(let k=0;k<4;k++) {
      const f=v=>cubicEdge(t,cs.map((z,i)=>k===i?v:z));
      close(g.coeffGrad[k],-.9*derivative(f,cs[k]));
    }
  }
});
test('KAN layer sums all inputs, biases, and repeated gradient fan-in',()=>{
  const k=layer(2,2,[.1,.2,.3,.4,.2,.1,.4,.6,.3,.1,.1,.2,.5,.4,.3,.2],[.1,-.2]);
  const x=[.4,.6], upstream=[1.5,-.7];
  const y=kanLayerForward(k,x);
  close(y[0],.1+cubicEdge(.4,k.coeff.slice(0,4))+cubicEdge(.6,k.coeff.slice(4,8)));
  close(y[1],-.2+cubicEdge(.4,k.coeff.slice(8,12))+cubicEdge(.6,k.coeff.slice(12,16)));
  const grad=kanLayerVjp(k,x,upstream);
  const loss=z=>kanLayerForward(k,z).reduce((s,v,i)=>s+v*upstream[i],0);
  for(let i=0;i<2;i++)close(grad.inputGrad[i],derivative(v=>loss(x.map((z,j)=>j===i?v:z)),x[i]));
  for(let j=0;j<k.coeff.length;j++) {
    const f=v=>kanLayerForward({...k,coeff:k.coeff.map((z,i)=>i===j?v:z)},x)
      .reduce((s,z,i)=>s+z*upstream[i],0);
    close(grad.coeffGrad[j],derivative(f,k.coeff[j]));
  }
  assert.deepEqual(grad.biasGrad,upstream);
});
test('MFKAN high stage consumes low-fidelity estimate and returns explicit correction branches',()=>{
  const cfg=fixture(), x=[.25,.7];
  const got=mfkanForward(cfg,x);
  close(got.high,got.low+cfg.mix*got.linear+(1-cfg.mix)*got.nonlinear);
  const changed={...cfg,low:{...cfg.low,bias:[cfg.low.bias[0]+.05]}};
  assert.notEqual(mfkanForward(changed,x).high,got.high);
  assert(got.low>=0&&got.low<=1);
});
test('MFKAN high correction and low stage derivatives match finite differences',()=>{
  const cfg=fixture(),x=[.25,.7],g=mfkanVjp(cfg,x,.8,{freezeLow:false});
  const loss=(c,v=x)=>.8*mfkanForward(c,v).high;
  for(let i=0;i<2;i++)close(g.inputGrad[i],derivative(v=>loss(cfg,x.map((z,j)=>j===i?v:z)),x[i]));
  for(let i=0;i<cfg.highLinear.length;i++) {
    const w=cfg.highLinear[i];
    close(g.highLinearGrad[i],derivative(v=>loss({...cfg,highLinear:cfg.highLinear.map((z,j)=>j===i?v:z)}),w));
  }
  for(let i=0;i<cfg.low.coeff.length;i++) {
    const w=cfg.low.coeff[i];
    const f=v=>loss({...cfg,low:{...cfg.low,coeff:cfg.low.coeff.map((z,j)=>j===i?v:z)}});
    close(g.lowCoeffGrad[i],derivative(f,w));
  }
  for(let i=0;i<cfg.highNonlinear.coeff.length;i++) {
    const w=cfg.highNonlinear.coeff[i];
    const f=v=>loss({...cfg,highNonlinear:{...cfg.highNonlinear,coeff:cfg.highNonlinear.coeff.map((z,j)=>j===i?v:z)}});
    close(g.highNonlinearCoeffGrad[i],derivative(f,w));
  }
  close(g.mixGrad,derivative(v=>loss({...cfg,mix:v}),cfg.mix));
});
test('freeze low fidelity blocks its parameter updates but preserves input gradients',()=>{
  const cfg=fixture(), x=[.4,.6];
  const frozen=mfkanVjp(cfg,x,1,{freezeLow:true});
  const trainable=mfkanVjp(cfg,x,1,{freezeLow:false});
  assert(frozen.lowCoeffGrad.every(x=>x===0));
  assert.deepEqual(frozen.lowBiasGrad,[0]);
  assert.deepEqual(frozen.inputGrad,trainable.inputGrad);
  assert(trainable.lowCoeffGrad.some(x=>x!==0));
});
test('bounded gradient descent improves a paired high-fidelity toy dataset',()=>{
  const cfg=fixture();cfg.mix=.5;
  // In this constrained profile the low model already approximates a smooth
  // baseline; fit the high linear correction to known paired measurements.
  const data=[[.2,.3],[.4,.6],[.6,.4],[.8,.7]];
  const target=x=>{const low=kanLayerForward(cfg.low,x)[0];return low+.2*x[0]-.1*x[1]+.15*low+.05;};
  const loss=()=>data.reduce((s,x)=>s+(mfkanForward(cfg,x).high-target(x))**2,0)/data.length;
  const initial=loss();
  for(let step=0;step<350;step++) {
    const dw=Array(cfg.highLinear.length).fill(0), dc=Array(cfg.highNonlinear.coeff.length).fill(0);
    let db=0;
    for(const x of data) {
      const diff=mfkanForward(cfg,x).high-target(x);
      const g=mfkanVjp(cfg,x,2*diff/data.length,{freezeLow:true});
      g.highLinearGrad.forEach((v,i)=>dw[i]+=v);
      g.highNonlinearCoeffGrad.forEach((v,i)=>dc[i]+=v);
      db+=g.highNonlinearBiasGrad[0];
    }
    for(let i=0;i<dw.length;i++)cfg.highLinear[i]-=.12*dw[i];
    for(let i=0;i<dc.length;i++)cfg.highNonlinear.coeff[i]-=.12*dc[i];
    cfg.highNonlinear.bias[0]-=.12*db;
  }
  assert(loss()<initial/20,`loss did not improve: ${initial} -> ${loss()}`);
});
test('fail closed for shapes, domain, quota, invalid numerical values, and non-normalized low output',()=>{
  assert.throws(()=>cubicBasis(-.001),/\[0,1\]/);
  assert.throws(()=>cubicBasis(NaN),/bounded finite/);
  assert.throws(()=>cubicEdge(.5,[1,2,3]),/shape mismatch/);
  assert.throws(()=>cubicEdge(.5,[1,2,Infinity,4]),/bounded finite/);
  assert.throws(()=>kanLayerForward(layer(2,1,[1,2,3,4],[0]),[.1,.2]),/shape mismatch/);
  assert.throws(()=>kanLayerForward(layer(1,1,[1,2,3,4],[0]),[1.1]),/\[0,1\]/);
  assert.throws(()=>kanLayerForward(layer(1e8,1e8,[],[]),[]),/bounded profile/);
  const cfg=fixture();
  assert.throws(()=>mfkanForward({...cfg,mix:-.1},[.3,.3]),/\[0,1\]/);
  assert.throws(()=>mfkanForward({...cfg,low:{...cfg.low,bias:[1.2]}},[.3,.3]),/\[0,1\]/);
  assert.throws(()=>mfkanForward({...cfg,highNonlinear:{...cfg.highNonlinear,inputs:2}},[.3,.3]),/expects/);
});
