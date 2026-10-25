// HOST-ONLY independent Cox-de Boor reference for packed scalar KANLayer.
import test from 'node:test';
import assert from 'node:assert/strict';
const LIMIT=1e6;
function basis(k,i,p,x){
  if(p===0)return k[i]<=x&&x<k[i+1]?1:0;
  const a=k[i+p]-k[i],b=k[i+p+1]-k[i+1];
  return (a>0?(x-k[i])/a*basis(k,i,p-1,x):0)
   +(b>0?(k[i+p+1]-x)/b*basis(k,i+1,p-1,x):0);
}
function edge(k,c,x){
  if(x===k[c.length])return c.at(-1);
  return c.reduce((s,v,i)=>s+v*basis(k,i,3,x),0);
}
function layerInto(xs,k,cs,bias,I,O,C,out,scratch){
  if(!Number.isInteger(I)||!Number.isInteger(O)||!Number.isInteger(C)||
    I<1||I>32||O<1||O>32||C<4||C>256||C>Math.floor(65536/(I*O)))
    throw RangeError('DIM');
  if([xs,k,cs,bias,out,scratch].some(a=>!Array.isArray(a)))
    throw RangeError('TYPE');
  if(out===scratch||out===xs||scratch===xs||out===k||scratch===k||
    out===cs||scratch===cs||out===bias||scratch===bias)throw RangeError('ALIAS');
  if(xs.length!==I||bias.length!==O||out.length!==O||scratch.length!==O||
    cs.length!==I*O*C||k.length!==C+4)throw RangeError('SHAPE');
  if([...xs,...k,...cs,...bias].some(v=>typeof v!=='number'||!Number.isFinite(v)||Math.abs(v)>LIMIT))
    throw RangeError('VALUE');
  const lo=k[3],hi=k[C];
  if(!(lo<hi)||k.slice(0,4).some(v=>v!==lo)||k.slice(C).some(v=>v!==hi))
    throw RangeError('CLAMP');
  let repeats=1;
  for(let j=1;j<k.length;j++){
    if(k[j]<k[j-1])throw RangeError('UNSORTED');
    repeats=k[j]===k[j-1]?repeats+1:1;
    if(j>=4&&j<C&&repeats>=4)throw RangeError('MULTIPLICITY');
  }
  for(let j=4;j<C;j++)if(k[j]<=lo||k[j]>=hi)throw RangeError('INTERIOR');
  if(xs.some(x=>x<lo||x>hi))throw RangeError('DOMAIN');
  for(let o=0;o<O;o++){
    let value=bias[o];
    for(let i=0;i<I;i++){
      const start=(o*I+i)*C;
      const part=edge(k,cs.slice(start,start+C),xs[i]);
      if(!Number.isFinite(part)||Math.abs(part)>LIMIT)throw RangeError('EDGE');
      value+=part;
      if(!Number.isFinite(value)||Math.abs(value)>LIMIT)throw RangeError('FANIN');
    }
    scratch[o]=value;
  }
  for(let o=0;o<O;o++)out[o]=scratch[o];
}
const close=(a,b,eps=2e-12)=>assert.ok(Math.abs(a-b)<=eps,String(a)+' != '+String(b));
let seed=0x20261025;
const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32};
const fixture=()=>({
  xs:[.25,.75],k:[0,0,0,0,1,1,1,1],
  cs:[...Array(4).fill(.2),...Array(4).fill(.3),...Array(4).fill(.4),...Array(4).fill(.5)],
  bias:[.1,-.1],I:2,O:2,C:4
});
const run=(f,out,scratch)=>layerInto(f.xs,f.k,f.cs,f.bias,f.I,f.O,f.C,out,scratch);
test('packed 2-in 2-out known constants',()=>{
 const f=fixture(),out=[9,9];run(f,out,[0,0]);close(out[0],.6);close(out[1],.8);
});
test('2000 seeded randomized multi-span packed edge forward comparisons',()=>{
 for(let trial=0;trial<2000;trial++){
  const I=1+Math.floor(rand()*5),O=1+Math.floor(rand()*5),C=4+Math.floor(rand()*6);
  const interior=Array.from({length:C-4},()=>.01+.98*rand()).sort((a,b)=>a-b);
  const k=[0,0,0,0,...interior,1,1,1,1],xs=Array.from({length:I},()=>rand());
  const cs=Array.from({length:I*O*C},()=>2*rand()-1);
  const bias=Array.from({length:O},()=>2*rand()-1);
  const out=Array(O).fill(9);
  layerInto(xs,k,cs,bias,I,O,C,out,Array(O).fill(0));
  for(let o=0;o<O;o++){
   let wanted=bias[o];
   for(let i=0;i<I;i++)wanted+=edge(k,cs.slice((o*I+i)*C,(o*I+i+1)*C),xs[i]);
   close(out[o],wanted);
  }
 }
});
test('maximum 256-control edges are supported',()=>{
 const C=256,k=[0,0,0,0,...Array.from({length:252},(_,i)=>(i+1)/253),1,1,1,1];
 const out=[5,6];
 layerInto([.5],k,Array(C*2).fill(.25),[0,0],1,2,C,out,[0,0]);
 close(out[0],.25);close(out[1],.25);
});
test('dimension and product quotas fail with unchanged destination',()=>{
 const f=fixture();
 for(const [I,O,C] of [[0,2,4],[2,33,4],[2,2,257],[32,32,256]]){
  const out=[77,88];assert.throws(()=>layerInto(f.xs,f.k,f.cs,f.bias,I,O,C,out,[0,0]));
  assert.deepEqual(out,[77,88]);
 }
});
test('invalid knots shapes NaN and out-of-domain reject atomically',()=>{
 const f=fixture();
 for(const bad of [
  {...f,cs:f.cs.slice(1)}, {...f,k:[0,0,0,0,.2,1,1,1]},
  {...f,xs:[.25,1.01]}, {...f,cs:f.cs.map((v,i)=>i===3?NaN:v)},
  {...f,bias:[Infinity,-.1]}
 ]){
  const out=[77,88];assert.throws(()=>run(bad,out,[0,0]));assert.deepEqual(out,[77,88]);
 }
});
test('overflow in later output does not partially publish earlier output',()=>{
 const f=fixture(),out=[77,88],scratch=[0,0];
 const cs=[...f.cs.slice(0,8),...Array(8).fill(1e6)];
 assert.throws(()=>run({...f,cs},out,scratch),/FANIN/);
 assert.deepEqual(out,[77,88]);close(scratch[0],.6);
});
test('output aliases input or scratch reject',()=>{
 const f=fixture(),out=[77,88];
 assert.throws(()=>run(f,out,out),/ALIAS/);assert.deepEqual(out,[77,88]);
 assert.throws(()=>run(f,f.xs,[0,0]),/ALIAS/);
});
test('triple interior knot is accepted on shared grid',()=>{
 const C=7,k=[0,0,0,0,.5,.5,.5,1,1,1,1],out=[0];
 layerInto([.5,.25],k,Array(C*2).fill(.125),[.25],2,1,C,out,[0]);
 close(out[0],.5);
});
test('both clamped endpoints are admitted',()=>{
 const f=fixture(),out=[0,0];run({...f,xs:[0,1]},out,[0,0]);
 close(out[0],.6);close(out[1],.8);
});
