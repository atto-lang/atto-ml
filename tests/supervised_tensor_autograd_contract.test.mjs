// HOST-ONLY independent numerical and source-composition oracle.
// This suite neither invokes the .ores compiler nor proves actor-state safety.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../examples/train_supervised_tensor_autograd.ores", import.meta.url),"utf8");

// These references are written independently rather than translated from
// internal Oreslang source or copied from the tensor/autograd repositories.
function pack(x,y,xvalid,yvalid,capacity) {
  if(x.length!==y.length||x.length!==xvalid.length||x.length!==yvalid.length)
    throw Error("ROW_SHAPE");
  const result={x:[],y:[],rows:[]};
  for(let i=0;i<x.length;i++){
    if(!xvalid[i]||!yvalid[i])continue;
    if(!Number.isFinite(x[i])||!Number.isFinite(y[i]))throw Error("NONFINITE");
    if(result.x.length>=capacity)throw Error("CAPACITY");
    result.x.push(x[i]);result.y.push(y[i]);result.rows.push(i);
  }
  return result;
}
function tape(w,x,b,y){
  const v=[w,x,b,-y,0,0,0,0];
  v[4]=v[0]*v[1];v[5]=v[4]+v[2];v[6]=v[5]+v[3];v[7]=v[6]*v[6];
  const d=Array(8).fill(0);d[7]=1;
  for(let i=7;i>=4;i--){
    if(i===7){d[6]+=d[7]*v[6];d[6]+=d[7]*v[6];}
    if(i===6){d[5]+=d[6];d[3]+=d[6];}
    if(i===5){d[4]+=d[5];d[2]+=d[5];}
    if(i===4){d[0]+=d[4]*v[1];d[1]+=d[4]*v[0];}
  }
  return {prediction:v[5],loss:v[7],dw:d[0],db:d[2]};
}
function train(x,y,epochs=500,rate=.05) {
  let w=0,b=0,last=Infinity,perSample=null;
  for(let epoch=0;epoch<epochs;epoch++){
    let dw=0,db=0,loss=0; const dws=[];
    for(let i=0;i<x.length;i++){
      const stage=tape(w,x[i],b,y[i]);
      if(stage.prediction!==w*x[i]+b)throw Error("FORWARD_DISAGREEMENT");
      dw+=stage.dw;db+=stage.db;loss+=stage.loss;
      dws.push(stage.dw);
    }
    if(epoch===0)perSample=dws;
    last=loss/x.length;
    const nw=w-rate*dw/x.length;
    const nb=b-rate*db/x.length;
    if(!Number.isFinite(nw)||!Number.isFinite(nb))throw Error("NONFINITE_STEP");
    w=nw;b=nb;
  }
  return {w,b,last,perSample};
}
function scatter(grad,rows,n){
  const out=Array(n).fill(0);
  for(let i=0;i<grad.length;i++)out[rows[i]]+=grad[i];
  return out;
}
test("source composes actual data, tensor, and autograd modules",()=>{
  assert.match(source,/compact_supervised_f64_into from "\.\.\/\.\.\/oreslang-data-science\/src\/pipeline\/ml_batch_bridge"/);
  assert.match(source,/matmul_2d_f64_into from "\.\.\/\.\.\/oreslang-tensor\/src\/tensor"/);
  assert.match(source,/forward_checked_into from "\.\.\/\.\.\/oreslang-autograd\/src\/forward_checked"/);
  assert.match(source,/backward_checked_into from "\.\.\/\.\.\/oreslang-autograd\/src\/backward_checked"/);
  assert.match(source,/ORES_ML_SUPERVISED_TENSOR_AUTOGRAD_PASS/);
  assert.doesNotMatch(source,/\bself\.state\b|\bdefine actor\b|\bpython\b/i);
});
test("training fixture pins a sole bad-source row and four valid rows",()=>{
  assert.match(source,/arr\[0\.0, 1\.0, 2\.0, 3\.0, 1000\.0\]/);
  assert.match(source,/arr\[1\.0, 3\.0, 5\.0, 7\.0, 2000\.0\]/);
  assert.match(source,/arr\[true, true, true, true, false\]/);
  assert.match(source,/Assert\.equal_throws\(batch\.count, 4\)/);
});
test("compacted mask intersection preserves both numeric values and row IDs",()=>{
  const got=pack([0,1,2,3,1000],[1,3,5,7,2000],
    [true,true,true,true,false],[true,true,true,true,false],4);
  assert.deepEqual(got,{x:[0,1,2,3],y:[1,3,5,7],rows:[0,1,2,3]});
});
test("unselected nonfinite rows remain excluded",()=>{
  const got=pack([1,NaN,3],[3,Infinity,7],[true,false,true],[true,true,true],2);
  assert.deepEqual(got,{x:[1,3],y:[3,7],rows:[0,2]});
});
test("mask shape, capacity and selected nonfinite reject",()=>{
  assert.throws(()=>pack([1,2],[3],[true,true],[true],2),/ROW_SHAPE/);
  assert.throws(()=>pack([1,2],[3,4],[true,true],[true,true],1),/CAPACITY/);
  assert.throws(()=>pack([1,Infinity],[3,4],[true,true],[true,true],2),/NONFINITE/);
});
test("the first SGD step is derived through a reverse-mode tape",()=>{
  const t=[0,1,2,3].map((x,i)=>tape(0,x,0,[1,3,5,7][i]));
  assert.deepEqual(t.map(v=>v.loss),[1,9,25,49]);
  assert.deepEqual(t.map(v=>v.dw),[0,-6,-20,-42]);
  assert.deepEqual(t.map(v=>v.db),[-2,-6,-10,-14]);
  assert.equal(t.reduce((s,x)=>s+x.dw,0),-68);
});
test("SGD improves actual supervised reference loss and converges",()=>{
  const result=train([0,1,2,3],[1,3,5,7]);
  assert(result.last<1e-6);
  assert(Math.abs(result.w-2)<.01);
  assert(Math.abs(result.b-1)<.01);
  assert(Number.isFinite(result.last));
});
test("invalid source row receives no backprop contribution",()=>{
  const result=train([0,1,2,3],[1,3,5,7]);
  const scattered=scatter(result.perSample,[0,1,2,3],5);
  assert.deepEqual(scattered,[0,-6,-20,-42,0]);
});
test("duplicate indexed reverse contributions accumulate",()=>{
  assert.deepEqual(scatter([1,3,5,7],[0,2,2,3],5),[1,0,8,7,0]);
});
test("tape VJP for weight and bias matches central finite differences",()=>{
  for(const w of [-2,0,1,4])for(const b of [-1,0,2])for(const x of [-3,0,2,4]) {
    const y=2*x+1,h=1e-5;
    const at=tape(w,x,b,y);
    const dw=(tape(w+h,x,b,y).loss-tape(w-h,x,b,y).loss)/(2*h);
    const db=(tape(w,x,b+h,y).loss-tape(w,x,b-h,y).loss)/(2*h);
    assert(Math.abs(at.dw-dw)<1e-7);
    assert(Math.abs(at.db-db)<1e-7);
  }
});
test("no parameter mutation occurs during gradient collection in source",()=>{
  const start=source.indexOf("for int epoch");
  const finish=source.indexOf("final_loss = batch_loss",start);
  const body=source.slice(start,finish);
  assert(body.length>500);
  assert.doesNotMatch(body,/weight\[0\]\s*=|bias\[0\]\s*=/);
  assert.match(source,/val float next_weight =/);
  assert.match(source,/val float next_bias =/);
});
