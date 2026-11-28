// Host-only independent numerical/reference gate for the proposed Oreslang
// 13-node two-layer ReLU reverse-mode guest fixture. NOT guest execution.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../examples/train_nullable_two_layer_autograd.ores",import.meta.url),"utf8");
const OP=[0,0,0,0,0,0,2,1,3,2,1,1,2];
const L=[-1,-1,-1,-1,-1,-1,0,6,7,3,9,10,11];
const R=[-1,-1,-1,-1,-1,-1,1,2,-1,8,4,5,11];
const maskBatch=(xs,ys,fx,fy)=>{
  if(xs.length!==ys.length||xs.length!==fx.length||xs.length!==fy.length)
    throw Error("BAD_BATCH_SHAPE");
  const result=[];
  for(let i=0;i<xs.length;i++)if(fx[i]&&fy[i]){
    if(!Number.isFinite(xs[i])||!Number.isFinite(ys[i]))throw Error("NONFINITE");
    result.push({row:i,x:xs[i],y:ys[i]});
  }
  return result;
};
function evalTape(w,x,y) {
  const v=[w[0],x,w[1],w[2],w[3],-y,...Array(7).fill(0)];
  if(v.length!==13)throw Error("TAPE_ARITY");
  for(let i=0;i<OP.length;i++){
    const op=OP[i],a=L[i],b=R[i];
    if(op===0){if(a!==-1||b!==-1)throw Error("BAD_LEAF");}
    else{
      if(a<0||a>=i||(op!==3&&(b<0||b>=i)))throw Error("BAD_DAG");
      if(op===1)v[i]=v[a]+v[b];
      else if(op===2)v[i]=v[a]*v[b];
      else if(op===3)v[i]=Math.max(0,v[a]);
      else throw Error("BAD_OP");
    }
    if(!Number.isFinite(v[i]))throw Error("NONFINITE");
  }
  const g=Array(13).fill(0);g[12]=1;
  for(let i=12;i>=0;i--){
    const op=OP[i],a=L[i],b=R[i],up=g[i];
    if(op===1){g[a]+=up;g[b]+=up;}
    if(op===2){g[a]+=up*v[b];g[b]+=up*v[a];}
    if(op===3&&v[a]>0)g[a]+=up;
  }
  if(g.some(x=>!Number.isFinite(x)))throw Error("NONFINITE_GRAD");
  return {loss:v[12],grad:[g[0],g[2],g[3],g[4]],inputGrad:g[1],values:v};
}
function train(rows,epochs=1000,rate=.02){
  if(!rows.length)throw Error("EMPTY");
  let parameters=[1,1,1,0],last=0,lastInputGrads=[];
  for(let e=0;e<epochs;e++){
    const grads=[0,0,0,0];let loss=0;
    for(const {x,y} of rows){
      const result=evalTape(parameters,x,y);loss+=result.loss;
      if(e===epochs-1)lastInputGrads.push(result.inputGrad);
      result.grad.forEach((g,i)=>grads[i]+=g);
    }
    const candidates=parameters.map((w,i)=>w-rate*grads[i]/rows.length);
    if(candidates.some(x=>!Number.isFinite(x))||!Number.isFinite(loss))
      throw Error("UNSAFE_COMMIT");
    parameters=candidates;last=loss/rows.length;
  }
  return {parameters,loss:last,lastInputGrads};
}
test("guest fixture imports real nullable data and actual Oreslang tape kernels",()=>{
  assert.match(source,/oreslang-data-science\/src\/pipeline\/ml_batch_bridge/);
  assert.match(source,/oreslang-autograd\/src\/tape/);
  assert.match(source,/forward_checked_into\(/);
  assert.match(source,/backward_checked_into\(/);
  assert.match(source,/ORES_TWO_LAYER_AUTOGRAD_PASS/);
});
test("fixture declares exact 13 node topological ReLU graph with square",()=>{
  assert.match(source,/arr\[0,0,0,0,0,0,2,1,3,2,1,1,2\]/);
  assert.match(source,/arr\[-1,-1,-1,-1,-1,-1,0,6,7,3,9,10,11\]/);
  assert.match(source,/arr\[-1,-1,-1,-1,-1,-1,1,2,-1,8,4,5,11\]/);
  assert(OP.length===13&&L.length===13&&R.length===13);
  assert.equal(OP[8],3);
  assert.equal(L[12],R[12]); // square VJP must accumulate twice
});
test("null data preserves provenance and cannot poison the tape",()=>{
  const rows=maskBatch([0,NaN,1,2,3],[1,Infinity,3,5,7],
    [true,false,true,true,true],[true,false,true,true,true]);
  assert.deepEqual(rows.map(x=>x.row),[0,2,3,4]);
  assert.deepEqual(rows.map(x=>x.y),[1,3,5,7]);
});
test("invalid selected data fails closed",()=>{
  assert.throws(()=>maskBatch([NaN],[3],[true],[true]),/NONFINITE/);
  assert.throws(()=>maskBatch([1,2],[3],[true,true],[true]),/BAD_BATCH_SHAPE/);
  assert.throws(()=>train([],1),/EMPTY/);
});
test("forward graph computes expected initial prediction and loss",()=>{
  const out=evalTape([1,1,1,0],2,5);
  assert.equal(out.values[7],3);
  assert.equal(out.values[8],3);
  assert.equal(out.values[10],3);
  assert.equal(out.values[11],-2);
  assert.equal(out.loss,4);
});
test("reverse-mode 4 parameter VJPs match independent central differences",()=>{
  const p=[1.2,0.9,1.3,-.1],h=1e-6;
  for(const x of [0,1,2,3])for(const y of [1,3,5,7]){
    const got=evalTape(p,x,y);
    for(let i=0;i<4;i++){
      const hi=p.slice(),lo=p.slice();hi[i]+=h;lo[i]-=h;
      const numeric=(evalTape(hi,x,y).loss-evalTape(lo,x,y).loss)/(2*h);
      assert(Math.abs(numeric-got.grad[i])<2e-7);
    }
  }
});
test("ReLU dead region uses zero hidden gradient",()=>{
  const r=evalTape([1,-4,1,0],2,5);
  assert.equal(r.values[8],0);
  assert.equal(r.grad[0],0);
  assert.equal(r.grad[1],0);
  assert.equal(r.grad[2],0);
});
test("full nullable two-layer CPU model converges in 1000 steps",()=>{
  const rows=maskBatch([0,Infinity,1,2,3],[1,NaN,3,5,7],
    [true,false,true,true,true],[true,false,true,true,true]);
  const r=train(rows);
  assert(r.loss<1e-6);
  for(const {x,y} of rows){
    const z=Math.max(0,r.parameters[0]*x+r.parameters[1]);
    const predicted=r.parameters[2]*z+r.parameters[3];
    assert(Math.abs(predicted-y)<1e-4);
  }
});
test("candidate validation prevents four-way partial publication",()=>{
  const state=[1,2,3,4],before=state.slice();
  const gradients=[-1e308,1,2,3],rate=8;
  const candidates=state.map((v,i)=>v-rate*gradients[i]);
  assert(candidates.some(v=>!Number.isFinite(v)));
  if(candidates.every(Number.isFinite))state.splice(0,4,...candidates);
  assert.deepEqual(state,before);
  assert.match(source,/val float cw1/);
  assert.match(source,/val float cb1/);
  assert.match(source,/val float cw2/);
  assert.match(source,/val float cb2/);
  const check=source.indexOf("if (cw1-cw1) != 0.0");
  const publish=source.indexOf("first_weight[0] = cw1");
  assert(check>0&&publish>check);
});
test("source does not pretend sequential tape is an actor-owned state capability",()=>{
  assert.doesNotMatch(source,/\bdefine actor\b/);
  assert.doesNotMatch(source,/\bself\.state\b/);
  assert.match(source,/Actor ownership is another milestone/);
});

test("rank-2 tensor gather and saved-forward VJP imports are real",()=>{
  assert.match(source,/oreslang-tensor\/src\/tensor/);
  assert.match(source,/oreslang-tensor\/src\/row_batches/);
  assert.match(source,/oreslang-autograd\/src\/row_gather_vjp/);
  assert.match(source,/gather_rows_2d_f64_into\(original,selected,gathered\)/);
  assert.match(source,/row_gather_vjp_with_saved_f64_checked_into\(/);
  assert.match(source,/saved_input_gradient\[i\] = gradient\[1\]/);
});
test("tape gradients w.r.t. source inputs agree with finite difference",()=>{
  const params=[1.1,1.3,1.2,-0.5],eps=1e-6;
  for(const x of [0,1,2,3])for(const y of [1,3,5,7]){
    const got=evalTape(params,x,y);
    const finite=(evalTape(params,x+eps,y).loss-evalTape(params,x-eps,y).loss)/(2*eps);
    assert(Math.abs(finite-got.inputGrad)<3e-7);
  }
});
test("final tape gradients scatter to original row IDs; null row receives zero",()=>{
  const batch=maskBatch([0,999,1,2,3],[1,999,3,5,7],
     [true,false,true,true,true],[true,false,true,true,true]);
  const result=train(batch);
  const restored=Array(5).fill(0);
  for(let i=0;i<batch.length;i++)restored[batch[i].row]+=result.lastInputGrads[i];
  assert.equal(restored[1],0);
  assert.deepEqual([restored[0],restored[2],restored[3],restored[4]],
    result.lastInputGrads);
});
test("repeated row indices must scatter-add rather than overwrite",()=>{
  const indices=[0,2,2,4],upstream=[1,3,5,7],result=Array(5).fill(0);
  for(let i=0;i<indices.length;i++)result[indices[i]]+=upstream[i];
  assert.deepEqual(result,[1,0,8,0,7]);
});
test("tensor source checks excluded row and all four reverse-mapped rows",()=>{
  assert.match(source,/reverse_gradient_output\[1\] != 0\.0/);
  for(const pair of [[0,0],[2,1],[3,2],[4,3]]) {
    const re=new RegExp("reverse_gradient_output\\["+pair[0]+"\\] != saved_input_gradient\\["+pair[1]+"\\]");
    assert.match(source,re);
  }
});


test("excluded NaN/Infinity source slots are sanitized before constructing TensorF64",()=>{
  const raw=[0,Infinity,1,2,3],targets=[1,NaN,3,5,7];
  const rows=maskBatch(raw,targets,
    [true,false,true,true,true],[true,false,true,true,true]);
  const projected=Array(raw.length).fill(0);
  for(const row of rows)projected[row.row]=row.x;
  assert.deepEqual(projected,[0,0,1,2,3]);
  assert(projected.every(Number.isFinite));
  assert(!raw.every(Number.isFinite));
  assert.match(source,/val List<float> projected_values = arr\[0\.0,0\.0,0\.0,0\.0,0\.0\]/);
  assert.match(source,/projected_values\[row_ids\[j\]\] = x\[j\]/);
  assert.match(source,/new TensorF64\(arr\[5,1\],projected_values\)/);
  assert.doesNotMatch(source,/new TensorF64\(arr\[5,1\],feature\.values\)/);
});
test("excluded feature-valid but target-null records also cannot poison tensor materialization",()=>{
  const raw=[1,NaN,3],targets=[2,999,6];
  const rows=maskBatch(raw,targets,[true,true,true],[true,false,true]);
  const projected=Array(raw.length).fill(0);
  for(const {row,x} of rows)projected[row]=x;
  assert.deepEqual(projected,[1,0,3]);
  assert(rows.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)));
});
