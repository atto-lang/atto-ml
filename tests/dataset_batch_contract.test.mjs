// HOST-ONLY deterministic aligned minibatch oracle, not compiled Oreslang.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../src/dataset_batches.ores",import.meta.url),"utf8");
const guest=readFileSync(new URL("../examples/train_aligned_dataset.ores",import.meta.url),"utf8");
let seed=0xc0ffee35;
function rand(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return (seed>>>0)/4294967296;}
function randint(n){return Math.floor(rand()*n);}
function tensor(rows,cols,data) {
  assert(Number.isSafeInteger(rows)&&rows>=0);
  assert(Number.isSafeInteger(cols)&&cols>=0);
  assert.equal(rows*cols,data.length);
  assert(data.every(Number.isFinite));
  return {rows,cols,data:[...data]};
}
function gather(t,indices) {
  const out=[];
  for(const id of indices) {
    if(!Number.isSafeInteger(id)||id<0||id>=t.rows)throw Error("BAD_INDEX");
    for(let c=0;c<t.cols;c++)out.push(t.data[id*t.cols+c]);
  }
  return out;
}
function aligned(features,labels,indices,fxScratch,fyScratch,fxOut,fyOut,injectFailure=false){
  // Host model enforces the same preflight and transactional publication intent.
  if(features.rows!==labels.rows)throw Error("ROW_MISMATCH");
  if(new Set([fxScratch,fyScratch,fxOut,fyOut]).size!==4)
    throw Error("OUTPUT_ALIAS");
  if([fxScratch,fyScratch,fxOut,fyOut].some(a=>a===features.data||a===labels.data))
    throw Error("SOURCE_ALIAS");
  if(fxScratch.length!==indices.length*features.cols||
     fxOut.length!==indices.length*features.cols||
     fyScratch.length!==indices.length*labels.cols||
     fyOut.length!==indices.length*labels.cols)
    throw Error("BAD_OUTPUT_EXTENT");
  for(const id of indices)if(!Number.isSafeInteger(id)||id<0||id>=features.rows)
    throw Error("BAD_INDEX");
  const x=gather(features,indices);
  for(let i=0;i<x.length;i++)fxScratch[i]=x[i];
  if(injectFailure)throw Error("SIMULATED_SECOND_GATHER_FAILURE");
  const y=gather(labels,indices);
  for(let i=0;i<y.length;i++)fyScratch[i]=y[i];
  for(let i=0;i<x.length;i++)fxOut[i]=fxScratch[i];
  for(let i=0;i<y.length;i++)fyOut[i]=fyScratch[i];
}
function train(x,y,steps=500,lr=.05){
  let w=0,b=0,loss=0;
  for(let k=0;k<steps;k++){
    let dw=0,db=0,total=0;
    for(let i=0;i<x.length;i++){
      const e=(w*x[i]+b)-y[i];
      total+=e*e;dw+=e*x[i];db+=e;
    }
    const scale=2/x.length;
    w-=lr*scale*dw;b-=lr*scale*db;
    loss=total/x.length;
  }
  return {w,b,loss};
}

test("source uses tensor gather twice with separate output scratch, after all preflights",()=>{
  assert.match(source,/pub fnc gather_supervised_batch_f64_checked_into/);
  assert.match(source,/RowBatches\.gather_rows_2d_f64_into\(rt borrow features, rt borrow indices, rt mut borrow feature_scratch\)/);
  assert.match(source,/RowBatches\.gather_rows_2d_f64_into\(rt borrow labels, rt borrow indices, rt mut borrow label_scratch\)/);
  assert.match(source,/feature_output\[i\] = feature_scratch\[i\]/);
  assert.match(source,/label_output\[i\] = label_scratch\[i\]/);
  const second=source.indexOf("RowBatches.gather_rows_2d_f64_into(rt borrow labels");
  const firstCommit=source.indexOf("feature_output[i] = feature_scratch[i]");
  assert(firstCommit>second);
  assert.match(guest,/ORES_ML_ALIGNED_DATASET_TRAINING_PASS/);
});

test("row order, duplicates and feature/label alignment are preserved",()=>{
  const f=tensor(4,2,[0,1,2,3,4,5,6,7]);
  const y=tensor(4,1,[1,5,9,13]);
  const a=Array(6).fill(0),b=Array(3).fill(0),c=Array(6).fill(0),d=Array(3).fill(0);
  aligned(f,y,[3,1,3],a,b,c,d);
  assert.deepEqual(c,[6,7,2,3,6,7]);
  assert.deepEqual(d,[13,5,13]);
});

test("late second-gather failure cannot publish either public output",()=>{
  const f=tensor(4,1,[0,1,2,3]),y=tensor(4,1,[1,3,5,7]);
  const a=Array(2).fill(0),b=Array(2).fill(0),c=Array(2).fill(99),d=Array(2).fill(88);
  assert.throws(()=>aligned(f,y,[1,3],a,b,c,d,true),/SIMULATED_SECOND_GATHER_FAILURE/);
  assert.deepEqual(c,[99,99]);
  assert.deepEqual(d,[88,88]);
  assert.deepEqual(a,[1,3]); // scratch is allowed to change
});

test("negative index, mismatched rows, scratch lengths and aliases fail",()=>{
  const f=tensor(3,1,[0,1,2]),y=tensor(3,1,[1,3,5]);
  for(const ids of [[3,1],[-1,0],[1,1.5]]) {
    const out=[99,99],lab=[88,88];
    assert.throws(()=>aligned(f,y,ids,[0,0],[0,0],out,lab),/BAD_INDEX/);
    assert.deepEqual(out,[99,99]);assert.deepEqual(lab,[88,88]);
  }
  assert.throws(()=>aligned(f,tensor(2,1,[1,3]),[1],[0],[0],[0],[0]),/ROW_MISMATCH/);
  const shared=[0];
  assert.throws(()=>aligned(f,y,[0],shared,[0],shared,[0]),/OUTPUT_ALIAS/);
  assert.throws(()=>aligned(f,y,[0],[0,0],[0],[0],[0]),/BAD_OUTPUT_EXTENT/);
});

test("empty paired datasets are legitimate and don't divide by zero",()=>{
  const f=tensor(0,3,[]),y=tensor(0,1,[]);
  const a=[],b=[],c=[],d=[];
  aligned(f,y,[],a,b,c,d);
  assert.deepEqual(c,[]);assert.deepEqual(d,[]);
});

test("seeded randomly selected batches preserve alignment and output invariants",()=>{
  for(let k=0;k<300;k++){
    const rows=randint(9),p=randint(6),q=randint(5),count=rows?randint(11):0;
    const f=tensor(rows,p,Array.from({length:rows*p},()=>rand()*3));
    const y=tensor(rows,q,Array.from({length:rows*q},()=>rand()*3));
    const ids=Array.from({length:count},()=>randint(rows));
    const a=Array(count*p).fill(-100),b=Array(count*q).fill(-100);
    const c=Array(count*p).fill(99),d=Array(count*q).fill(99);
    aligned(f,y,ids,a,b,c,d);
    assert.deepEqual(c,gather(f,ids));
    assert.deepEqual(d,gather(y,ids));
    assert.deepEqual(f.data.length,rows*p);
  }
});

test("aligned repeated-row minibatch drives loss-decreasing affine SGD",()=>{
  const f=tensor(4,1,[0,1,2,3]),y=tensor(4,1,[1,3,5,7]);
  const x=Array(3).fill(0),labels=Array(3).fill(0);
  aligned(f,y,[3,1,3],Array(3).fill(0),Array(3).fill(0),x,labels);
  const trained=train(x,labels);
  assert(Math.abs(trained.w-2)<.01);
  assert(Math.abs(trained.b-1)<.01);
  assert(trained.loss<1e-6);
});
