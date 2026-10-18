// HOST-ONLY independent cross-repo acceptance oracle. This does not run .ores.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../examples/train_nullable_data_with_autograd.ores",import.meta.url),"utf8");
function prepare(features,targets,fx,ty) {
  if(features.length!==targets.length||features.length!==fx.length||
     targets.length!==ty.length)throw Error("SHAPE");
  const rows=[];
  for(let i=0;i<fx.length;i++)if(fx[i]&&ty[i]){
    if(!Number.isFinite(features[i])||!Number.isFinite(targets[i]))
      throw Error("NONFINITE");
    rows.push(Object.freeze({index:i,x:features[i],y:targets[i]}));
  }
  return rows;
}
function train(data,epochs=500,rate=.05) {
  let w=0,b=0,lastLoss=0;
  for(let epoch=0;epoch<epochs;epoch++){
    let dw=0,db=0,sse=0;
    for(const {x,y} of data){
      const err=w*x+b-y;
      sse+=err*err;dw+=2*x*err;db+=2*err;
    }
    lastLoss=sse/data.length;
    w-=rate*dw/data.length;
    b-=rate*db/data.length;
  }
  return {w,b,lastLoss};
}
test("guest source actually imports both data-science and autograd",()=>{
  assert.match(source,/oreslang-data-science\/src\/pipeline\/ml_batch_bridge/);
  assert.match(source,/oreslang-autograd\/src\/forward_checked/);
  assert.match(source,/oreslang-autograd\/src\/backward_checked/);
  assert.match(source,/compact_supervised_f64_into\(/);
  assert.match(source,/forward_checked_into\(/);
  assert.match(source,/backward_checked_into\(/);
  assert.match(source,/ORES_DATA_AUTOGRAD_TRAINING_PASS/);
});
test("nullable supervised batching recovers four training rows",()=>{
  const rows=prepare([0,999,1,2,3],[1,999,3,5,7],
    [true,false,true,true,true],[true,false,true,true,true]);
  assert.deepEqual(rows.map(x=>x.index),[0,2,3,4]);
  assert.deepEqual(rows.map(x=>x.y),[1,3,5,7]);
});
test("null rows cannot poison training weights",()=>{
  const rows=prepare([0,Infinity,1,2,3],[1,NaN,3,5,7],
    [true,false,true,true,true],[true,false,true,true,true]);
  const a=train(rows);
  assert(Math.abs(a.w-2)<.01);
  assert(Math.abs(a.b-1)<.01);
  assert(a.lastLoss<1e-6);
});
test("finite selected invalid values fail before training",()=>{
  assert.throws(()=>prepare([1,NaN],[2,4],[true,true],[true,true]),/NONFINITE/);
  assert.throws(()=>prepare([1,2],[2],[true,true],[true]),/SHAPE/);
});
test("numerical scalar gradients agree with independent finite differences",()=>{
  const data=prepare([0,1,2,3],[1,3,5,7],[true,true,true,true],[true,true,true,true]);
  const loss=(w,b)=>data.reduce((sum,p)=>sum+(w*p.x+b-p.y)**2,0)/data.length;
  const w=1.2,b=-.5,eps=1e-6;
  let dw=0,db=0;
  for(const p of data){let e=w*p.x+b-p.y;dw+=2*e*p.x;db+=2*e;}
  dw/=data.length;db/=data.length;
  assert(Math.abs(dw-(loss(w+eps,b)-loss(w-eps,b))/(2*eps))<1e-8);
  assert(Math.abs(db-(loss(w,b+eps)-loss(w,b-eps))/(2*eps))<1e-8);
});
test("full 500-step CPU reference converges to known affine parameters",()=>{
  const data=prepare([0,1,2,3],[1,3,5,7],[true,true,true,true],[true,true,true,true]);
  const r=train(data);
  assert(r.lastLoss<1e-6);
  assert(r.w>1.99&&r.w<2.01);
  assert(r.b>.99&&r.b<1.01);
});


test("source rejects nonfinite accumulated gradients before optimizer update",()=>{
  const w=source.indexOf("val float next_grad_weight");
  const g=source.indexOf("if (next_grad_weight - next_grad_weight)");
  const publish=source.indexOf("grad_weight = next_grad_weight");
  assert(w>=0 && g>w && publish>g);
  assert.match(source,/ORES_DATA_AUTOGRAD_FAIL accumulation/);
});
test("source stages BOTH candidates before publishing weight or bias",()=>{
  const w=source.indexOf("val float candidate_weight");
  const b=source.indexOf("val float candidate_bias");
  const check=source.indexOf("if (candidate_loss - candidate_loss)");
  const pw=source.indexOf("weight[0] = candidate_weight");
  const pb=source.indexOf("bias[0] = candidate_bias");
  assert(w>=0 && b>w && check>b && pw>check && pb>pw);
  assert.match(source,/candidate_weight - candidate_weight/);
  assert.match(source,/candidate_bias - candidate_bias/);
  assert.match(source,/ORES_DATA_AUTOGRAD_FAIL optimizer/);
});
test("independent two-parameter SGD rejects overflow without partial publication",()=>{
  const state={weight:1e308,bias:7};
  const before={...state};
  function guarded(gradientWeight,gradientBias,learningRate) {
    if(![gradientWeight,gradientBias,learningRate].every(Number.isFinite))
      throw Error("INPUT");
    const w=state.weight-learningRate*gradientWeight;
    const b=state.bias-learningRate*gradientBias;
    if(!Number.isFinite(w)||!Number.isFinite(b))throw Error("OVERFLOW");
    state.weight=w;state.bias=b;
  }
  assert.throws(()=>guarded(-1e308,2,8),/OVERFLOW/);
  assert.deepEqual(state,before);
  guarded(1,2,0.5);
  assert.equal(state.weight,1e308);
  assert.equal(state.bias,6);
});
test("independent gradient accumulator detects overflow before retaining contribution",()=>{
  const accumulator={dw:Number.MAX_VALUE,db:2,loss:1};
  const original={...accumulator};
  const proposed={
    dw:accumulator.dw+Number.MAX_VALUE,
    db:accumulator.db+1,
    loss:accumulator.loss+2
  };
  assert(!Number.isFinite(proposed.dw));
  if(Object.values(proposed).every(Number.isFinite))
    Object.assign(accumulator,proposed);
  assert.deepEqual(accumulator,original);
});
