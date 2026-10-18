// HOST-ONLY independent CPU mathematical oracle. It MUST NOT be described
// as executing .ores code, validating actor runtime, or owning a frozen tensor.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../examples/train_columnar_tensor_autograd_cpu.ores",import.meta.url),"utf8");
const OP=[0,0,0,0,0,0,2,1,3,2,1,1,2];
const L=[-1,-1,-1,-1,-1,-1,0,6,7,8,9,10,11];
const R=[-1,-1,-1,-1,-1,-1,4,1,-1,2,3,5,11];

function pack(x,y,validX,validY,capacity) {
  if(x.length!==y.length || x.length!==validX.length || x.length!==validY.length)
    throw Error("ROW_OR_MASK_SHAPE");
  if(!Number.isSafeInteger(capacity) || capacity<0)throw Error("CAPACITY");
  const features=[],labels=[],rows=[];
  for(let i=0;i<x.length;i++)if(validX[i]&&validY[i]) {
    if(!Number.isFinite(x[i])||!Number.isFinite(y[i]))throw Error("NONFINITE");
    if(features.length>=capacity)throw Error("BACKPRESSURE");
    features.push(x[i]);labels.push(y[i]);rows.push(i);
  }
  return {features,labels,rows,count:features.length};
}
function gather(values,indices) {
  return indices.map(i=> {
    if(!Number.isInteger(i)||i<0||i>=values.length)throw Error("INDEX");
    return values[i];
  });
}
function matmul2d(a,rows,inner,b,cols) {
  if(a.length!==rows*inner||b.length!==inner*cols)throw Error("SHAPE");
  const out=Array(rows*cols).fill(0);
  for(let i=0;i<rows;i++)for(let j=0;j<cols;j++)for(let k=0;k<inner;k++)
    out[i*cols+j]+=a[i*inner+k]*b[k*cols+j];
  return out;
}
function tape(p,x,y) {
  const v=[...p,x,-y,...Array(7).fill(0)];
  for(let i=6;i<OP.length;i++){
    const a=v[L[i]],b=R[i]<0?undefined:v[R[i]];
    v[i]=OP[i]===1?a+b:OP[i]===2?a*b:Math.max(0,a);
    if(!Number.isFinite(v[i]))throw Error("NONFINITE_FORWARD");
  }
  const g=Array(13).fill(0);g[12]=1;
  for(let i=12;i>=6;i--){
    const a=L[i],b=R[i],up=g[i];
    if(OP[i]===1){g[a]+=up;g[b]+=up;}
    if(OP[i]===2){g[a]+=up*v[b];g[b]+=up*v[a];}
    if(OP[i]===3&&v[a]>0)g[a]+=up;
  }
  return {loss:v[12],grad:g.slice(0,4),dx:g[4],saved:v};
}
function train(x,y,epochs=1000,rate=.03) {
  const p=[1,0,1,0],first=[];
  let last=0,upstream=[];
  for(let ep=0;ep<epochs;ep++){
    const gradient=[0,0,0,0];
    let total=0; upstream=[];
    for(let i=0;i<x.length;i++){
      const {loss,grad,dx}=tape(p,x[i],y[i]);
      total+=loss;upstream.push(dx);
      grad.forEach((v,j)=>gradient[j]+=v);
    }
    last=total/x.length;
    if(ep===0)first.push(last);
    p.forEach((_,j)=>p[j]-=rate*gradient[j]/x.length);
  }
  return {first:first[0],last,parameters:p,upstream};
}
function scatter(indices,grad,n) {
  if(indices.length!==grad.length)throw Error("SHAPE");
  const out=Array(n).fill(0);
  indices.forEach((row,i)=>{
    if(!Number.isInteger(row)||row<0||row>=n)throw Error("INDEX");
    out[row]+=grad[i];
    if(!Number.isFinite(out[row]))throw Error("OVERFLOW");
  });
  return out;
}
test("fixture directly imports four genuine sibling Oreslang library layers",()=>{
  assert.match(source,/compact_supervised_f64_into from "\.\.\/\.\.\/oreslang-data-science\/src\/pipeline\/ml_batch_bridge"/);
  assert.match(source,/matmul_2d_f64_into from "\.\.\/\.\.\/oreslang-tensor\/src\/tensor"/);
  assert.match(source,/gather_rows_2d_f64_into from "\.\.\/\.\.\/oreslang-tensor\/src\/row_batches"/);
  assert.match(source,/backward_checked_into from "\.\.\/\.\.\/oreslang-autograd\/src\/backward_checked"/);
  assert.match(source,/row_gather_vjp_with_saved_f64_checked_into/);
  assert.match(source,/ORES_ML_COLUMNAR_TENSOR_AUTOGRAD_CPU_PASS/);
});
test("every tensor and autograd buffer in the example has an explicit output",()=>{
  assert.match(source,/gather_rows_2d_f64_into\(source_tensor, row_indices, gathered_x\)/);
  assert.match(source,/gather_rows_2d_f64_into\(label_tensor, row_indices, gathered_y\)/);
  assert.match(source,/matmul_2d_f64_into\(input_tensor, identity, projected\)/);
  assert.match(source,/backward_checked_into\(/);
  assert.doesNotMatch(source,/\bself\.state\b/);
});
test("supervised null-mask intersection retains aligned features, targets, and provenance",()=>{
  const out=pack([-2,-1,0,1,2,777,888],[1,1,1,3,5,999,999],
    [true,true,true,true,true,false,true],
    [true,true,true,true,true,true,false],5);
  assert.deepEqual(out.features,[-2,-1,0,1,2]);
  assert.deepEqual(out.labels,[1,1,1,3,5]);
  assert.deepEqual(out.rows,[0,1,2,3,4]);
  assert.equal(out.count,5);
});
test("differing mask lengths, undersized output and selected nonfinite values are rejected",()=>{
  assert.throws(()=>pack([1],[2],[true],[true,false],1),/ROW_OR_MASK_SHAPE/);
  assert.throws(()=>pack([1,2],[2,4],[true,true],[true,true],1),/BACKPRESSURE/);
  assert.throws(()=>pack([Infinity],[2],[true],[true],1),/NONFINITE/);
  assert.deepEqual(pack([Infinity],[2],[false],[true],0).features,[]);
});
test("matmul identity is a checked tensor step over resampled rows",()=>{
  const src=[-2,-1,0,1,2],rows=[0,1,3,3,4],selected=gather(src,rows);
  assert.deepEqual(matmul2d(selected,5,1,[1],1),[-2,-1,1,1,2]);
  assert.throws(()=>matmul2d(selected,5,2,[1],1),/SHAPE/);
});
test("13-node DAG is valid, and ReLU at negative input uses zero gradient",()=>{
  for(let i=0;i<OP.length;i++)if(OP[i]!==0) {
    assert(L[i]>=0&&L[i]<i);
    if(OP[i]===3)assert.equal(R[i],-1);
    else assert(R[i]>=0&&R[i]<i);
  }
  const probe=tape([1,0,1,0],-2,1);
  assert.equal(probe.dx,0);
  assert.equal(probe.grad[0],0);
});
test("autograd parameter gradients match central finite differences away from ReLU kink",()=>{
  for(const x of [-3,-1.5,.5,2,5]){
    const p=[1.3,.2,1.1,.8],y=x<0?1:1+2*x,analytic=tape(p,x,y),eps=1e-5;
    for(let i=0;i<4;i++){
      const hi=p.slice(),lo=p.slice();hi[i]+=eps;lo[i]-=eps;
      const d=(tape(hi,x,y).loss-tape(lo,x,y).loss)/(2*eps);
      assert(Math.abs(d-analytic.grad[i])<1e-7);
    }
    const dx=(tape(p,x+eps,y).loss-tape(p,x-eps,y).loss)/(2*eps);
    assert(Math.abs(dx-analytic.dx)<1e-7);
  }
});
test("two-layer ReLU network trained on masked/resampled data converges",()=>{
  const x=[-2,-1,1,1,2],y=[1,1,3,3,5];
  const result=train(x,y);
  assert(result.first>1);
  assert(result.last<1e-6);
  assert(result.last<result.first);
  assert(result.parameters[3]>.99&&result.parameters[3]<1.01);
});
test("saved gather VJP accumulates repeated rows and agrees with finite differences",()=>{
  const x=[-2,-1,0,1,2],index=[0,1,3,3,4],upstream=[1,2,-3,5,4];
  const s=scatter(index,upstream,x.length);
  assert.deepEqual(s,[1,2,0,2,4]);
  const f=values=>index.reduce((acc,j,i)=>acc+values[j]*upstream[i],0);
  const eps=1e-6;
  for(let i=0;i<x.length;i++){
    const a=x.slice(),b=x.slice();a[i]+=eps;b[i]-=eps;
    assert(Math.abs((f(a)-f(b))/(2*eps)-s[i])<1e-8);
  }
});
test("randomized masked gathers conserve linear adjoint dot products",()=>{
  let seed=0x43215678;
  const next=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
  for(let trial=0;trial<256;trial++){
    const n=(next()%9)+1,m=next()%30;
    const x=Array.from({length:n},(_,i)=>i+1);
    const index=Array.from({length:m},()=>next()%n);
    const up=index.map((_,i)=>((next()%301)-150)/37);
    const g=scatter(index,up,n);
    const lhs=gather(x,index).reduce((sum,z,i)=>sum+z*up[i],0);
    const rhs=x.reduce((sum,z,i)=>sum+z*g[i],0);
    assert(Math.abs(lhs-rhs)<1e-8);
  }
});
