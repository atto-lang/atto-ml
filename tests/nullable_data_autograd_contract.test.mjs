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
