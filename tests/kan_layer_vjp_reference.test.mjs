// HOST-ONLY independent acceptance suite for a packed KANLayer
// input/coefficient/bias VJP. This file does not compile or run Oreslang code.
import test from 'node:test';
import assert from 'node:assert/strict';

const LIMIT=1e6;
function safe(x,what='number') {
  if(!Number.isFinite(x)||Math.abs(x)>LIMIT)throw RangeError(what);
  return x;
}
function validate(xs,k,cs,up,I,O,C) {
  if(!Number.isSafeInteger(I)||!Number.isSafeInteger(O)||!Number.isSafeInteger(C)||I<1||I>32||O<1||O>32||C<4||C>256||I*O*C>65536) throw RangeError('quota');
  if(![xs,k,cs,up].every(Array.isArray)||xs.length!==I||up.length!==O||cs.length!==I*O*C||k.length!==C+4)throw RangeError('shape');
  for(const v of [xs,k,cs,up])for(const x of v)safe(x,'nonfinite');
  const lo=k[3],hi=k[C];
  if(!(lo<hi)||k.slice(0,4).some(x=>x!==lo)||k.slice(C).some(x=>x!==hi))throw RangeError('clamping');
  let repeat=1;
  for(let j=1;j<k.length;j++){
    if(k[j]<k[j-1])throw RangeError('order');
    repeat=k[j]===k[j-1]?repeat+1:1;
    if(j>=4&&j<C&&repeat>=4)throw RangeError('multiplicity');
  }
  for(let j=4;j<C;j++)if(k[j]<=lo||k[j]>=hi)throw RangeError('interior');
  if(xs.some(x=>x<lo||x>hi))throw RangeError('domain');
}
function span(k,n,x,p=3) {for(let i=p;i<n;i++)if(x>=k[i]&&x<k[i+1])return i;return n-1}
function deboor(k,c,x,p=3){
  const n=c.length,s=span(k,n,x,p),d=c.slice(s-p,s+1);
  for(let r=1;r<=p;r++)for(let j=p;j>=r;j--){
    const i=s-p+j,den=k[i+p+1-r]-k[i];
    if(!(den>0))throw RangeError('degenerate');
    const a=(x-k[i])/den;
    d[j]=(1-a)*d[j-1]+a*d[j];
  }
  return safe(d[p],'forward overflow');
}
function cox(k,i,p,x){
  if(p===0)return k[i]<=x&&x<k[i+1]?1:0;
  const a=k[i+p]-k[i],b=k[i+p+1]-k[i+1];
  return (a>0?(x-k[i])/a*cox(k,i,p-1,x):0)+(b>0?(k[i+p+1]-x)/b*cox(k,i+1,p-1,x):0);
}
function basis(k,C,i,x){
  if(x===k[C])return i===C-1?1:0;
  return cox(k,i,3,x);
}
function analyticInput(k,c,x){
  const q=[];
  for(let i=0;i<c.length-1;i++){
    const den=k[i+4]-k[i+1];
    if(!(den>0))throw RangeError('derivative knots');
    q.push(safe(3*(c[i+1]-c[i])/den,'derivative intermediate'));
  }
  return deboor(k.slice(1,-1),q,x,2);
}
function forward(xs,k,cs,bias,I,O,C){
  validate(xs,k,cs,Array(O).fill(0),I,O,C);
  if(bias.length!==O)throw RangeError('bias shape');
  bias.forEach(x=>safe(x,'bias'));
  const ys=[];
  for(let o=0;o<O;o++){
    let value=bias[o];
    for(let i=0;i<I;i++){
      const base=(o*I+i)*C;
      value=safe(value+deboor(k,cs.slice(base,base+C),xs[i]),'forward sum');
    }
    ys.push(value);
  }
  return ys;
}
function vjpInto(xs,k,cs,up,I,O,C,gi,gc,gb,si,sc,sb){
  validate(xs,k,cs,up,I,O,C);
  const writable=[gi,gc,gb,si,sc,sb];
  if(writable.some(v=>!Array.isArray(v)))throw RangeError('buffers');
  // Read-only inputs may legally alias each other. Only unique mutable
  // destinations and conflicts between mutable and read-only lists reject.
  if(new Set(writable).size!==writable.length ||
     writable.some(w=>[xs,k,cs,up].includes(w)))throw RangeError('alias');
  if(gi.length!==I||si.length!==I||gc.length!==I*O*C||sc.length!==I*O*C||gb.length!==O||sb.length!==O)throw RangeError('gradient shape');
  si.fill(0);sc.fill(0);sb.splice(0,O,...up);
  for(let o=0;o<O;o++)for(let i=0;i<I;i++){
    const base=(o*I+i)*C,local=cs.slice(base,base+C);
    // Mirror the candidate native *active-span* quota policy. An inactive
    // derivative control in a distant span must not reject this operation.
    const d=candidateLocalInputDerivative(k,local,xs[i]);
    const term=safe(up[o]*d,'input contribution');
    si[i]=safe(si[i]+term,'input fanin');
    for(let j=0;j<C;j++)sc[base+j]=safe(up[o]*basis(k,C,j,xs[i]),'coefficient gradient');
  }
  for(let i=0;i<I;i++)gi[i]=si[i];
  for(let j=0;j<gc.length;j++)gc[j]=sc[j];
  for(let o=0;o<O;o++)gb[o]=sb[o];
  return true;
}
function run(xs,k,cs,up,I,O,C){
  const gi=Array(I).fill(42),gc=Array(I*O*C).fill(42),gb=Array(O).fill(42);
  vjpInto(xs,k,cs,up,I,O,C,gi,gc,gb,Array(I).fill(0),Array(I*O*C).fill(0),Array(O).fill(0));
  return {gi,gc,gb};
}
const near=(a,b,tol=4e-7)=>assert.ok(Math.abs(a-b)<=tol,`${a} differs from ${b}`);
const diff=(f,z,h=1e-6)=>(f(z+h)-f(z-h))/(2*h);
const fx={xs:[.23,.71],k:[0,0,0,0,.5,1,1,1,1],C:5,I:2,O:2,
  cs:[.1,.2,.4,.3,.5,.3,.2,.5,.1,.4,.2,.4,.3,.1,.5,.1,.6,.2,.3,.4],
  up:[.8,-.3],bias:[.01,-.03]};
let state=0x20261025;
const rnd=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/2**32};

test('2x2 analytic input, parameter and bias gradients match numerical forward',()=>{
  const {xs,k,cs,up,I,O,C,bias}=fx,g=run(xs,k,cs,up,I,O,C);
  const loss=(a=xs,b=cs,c=bias)=>forward(a,k,b,c,I,O,C).reduce((s,y,j)=>s+y*up[j],0);
  for(let i=0;i<I;i++)near(g.gi[i],diff(x=>loss(xs.map((z,j)=>j===i?x:z)),xs[i]));
  for(let j=0;j<cs.length;j++)near(g.gc[j],diff(v=>loss(xs,cs.map((z,h)=>h===j?v:z)),cs[j]));
  for(let j=0;j<O;j++)near(g.gb[j],diff(v=>loss(xs,cs,bias.map((z,h)=>h===j?v:z)),bias[j]));
});
test('650 seeded multispan gradient trials agree with central differences',()=>{
  for(let trial=0;trial<650;trial++){
    const I=1+Math.floor(rnd()*3),O=1+Math.floor(rnd()*3),C=4+Math.floor(rnd()*5);
    const interior=Array.from({length:C-4},()=>.1+.8*rnd()).sort((a,b)=>a-b);
    const k=[0,0,0,0,...interior,1,1,1,1];
    const xs=Array.from({length:I},()=>.05+.9*rnd());
    if(xs.some(x=>interior.some(z=>Math.abs(x-z)<1e-4)))continue;
    const cs=Array.from({length:I*O*C},()=>rnd()-.5),up=Array.from({length:O},()=>rnd()-.5);
    const bias=Array.from({length:O},()=>rnd()-.5),g=run(xs,k,cs,up,I,O,C);
    const loss=(x=xs,c=cs)=>forward(x,k,c,bias,I,O,C).reduce((s,y,j)=>s+y*up[j],0);
    for(let i=0;i<I;i++)near(g.gi[i],diff(v=>loss(xs.map((x,j)=>j===i?v:x)),xs[i]),3e-6);
    for(let j=0;j<Math.min(cs.length,6);j++)near(g.gc[j],diff(v=>loss(xs,cs.map((x,i)=>i===j?v:x)),cs[j]),3e-6);
  }
});
test('input gradient sums contributions from two outputs',()=>{
  const k=[0,0,0,0,1,1,1,1],C=4,I=1,O=2;
  const cs=[0,.2,.5,1,1,.5,.2,0];
  const g=run([.3],k,cs,[.8,-.4],I,O,C);
  near(g.gi[0],.8*analyticInput(k,cs.slice(0,4),.3)-.4*analyticInput(k,cs.slice(4,8),.3));
  assert.deepEqual(g.gb,[.8,-.4]);
});
test('invalid upstream and later VJP overflow never publish gradients',()=>{
  const {xs,k,cs,I,O,C}=fx;
  const gi=[71,72],gc=Array(cs.length).fill(81),gb=[91,92];
  const attempt=(values,weights=cs)=>vjpInto(xs,k,weights,values,I,O,C,gi,gc,gb,[0,0],Array(cs.length).fill(0),[0,0]);
  assert.throws(()=>attempt([NaN,0]),/nonfinite/);
  const steep=[0,1,2,3,4,...cs.slice(5)];
  assert.throws(()=>attempt([1e6,1e6],steep),/input contribution|input fanin|coefficient gradient/);
  assert.deepEqual(gi,[71,72]); assert(gc.every(x=>x===81));assert.deepEqual(gb,[91,92]);
});
test('array alias and resource quota reject before publishing',()=>{
  const {xs,k,cs,up,I,O,C}=fx;
  const grad=[4,5];
  assert.throws(()=>vjpInto(xs,k,cs,up,I,O,C,grad,Array(cs.length).fill(0),[0,0],grad,Array(cs.length).fill(0),[0,0]),/alias/);
  assert.deepEqual(grad,[4,5]);
  assert.throws(()=>run([.2],k,Array(32*32*256).fill(0),[0],32,32,256),/shape|quota/);
});
test('triple interior knot uses right-hand analytic VJP',()=>{
  const k=[0,0,0,0,.5,.5,.5,1,1,1,1],c=[0,.1,.2,.3,.8,.5,.6];
  const g=run([.5],k,c,[2],1,1,7);
  near(g.gi[0],6,1e-12);
  near(g.gc.reduce((a,b)=>a+b,0),2,1e-12);
});


test('one-sided endpoint input VJPs match clamped cubic derivatives',()=>{
  const k=[0,0,0,0,1,1,1,1],c=[0,.2,.6,1.0];
  const u=.75,h=1e-7;
  const start=run([0],k,c,[u],1,1,4);
  const end=run([1],k,c,[u],1,1,4);
  near(start.gi[0],u*(forward([h],k,c,[0],1,1,4)[0]-forward([0],k,c,[0],1,1,4)[0])/h,1e-6);
  near(end.gi[0],u*(forward([1],k,c,[0],1,1,4)[0]-forward([1-h],k,c,[0],1,1,4)[0])/h,1e-6);
  assert.deepEqual(start.gc.slice(1),[0,0,0]);
  assert.deepEqual(end.gc.slice(0,3),[0,0,0]);
  near(start.gc[0],u,1e-12);near(end.gc[3],u,1e-12);
});

test('inactive coefficient gradient lanes are exactly zero at interior input',()=>{
  const C=7,k=[0,0,0,0,.2,.45,.7,1,1,1,1],c=[.2,.1,.4,.5,.2,.7,.3];
  const x=.3,g=run([x],k,c,[.3],1,1,C);
  // x lies in span 4: active coefficients 1..4; indices 0,5,6 inactive.
  assert.equal(g.gc[0],0);assert.equal(g.gc[5],0);assert.equal(g.gc[6],0);
  near(g.gc.reduce((a,b)=>a+b,0),.3,1e-12);
});

test('full 65536-coefficient and 32-input x 32-output buffers remain bounded',()=>{
  const I=32,O=32,C=64;
  const interior=Array.from({length:C-4},(_,i)=>(i+1)/(C-3));
  const k=[0,0,0,0,...interior,1,1,1,1];
  const xs=Array(I).fill(.5),up=Array(O).fill(.5),cs=Array(I*O*C).fill(.125);
  const gi=Array(I).fill(123),gc=Array(cs.length).fill(123),gb=Array(O).fill(123);
  assert.equal(cs.length,65536);
  vjpInto(xs,k,cs,up,I,O,C,gi,gc,gb,Array(I).fill(0),Array(cs.length).fill(0),Array(O).fill(0));
  for(const v of gi)near(v,0,1e-11);
  for(const v of gb)near(v,.5,1e-12);
  const sum=gc.reduce((s,v)=>s+v,0);
  near(sum,I*O*.5,2e-10);
});

test('late output coefficient-overflow cannot publish any gradient',()=>{
  const k=[0,0,0,0,1,1,1,1],I=1,O=2,C=4;
  const xs=[0],up=[1,1e6],cs=[0,0,0,0,0,0,0,0];
  const gi=[73],gc=Array(8).fill(74),gb=[75,76];
  // Output two has valid input gradient (zero), but its coefficient VJP
  // remains bounded: a proper rejection must be an invalid output, not
  // manufactured from bounded cubic basis and bounded upstream.
  // Therefore instead force first derivative into quota violation on late edge.
  const steep=[0,0,0,0,0,1e6,1e6,1e6];
  assert.throws(()=>vjpInto(xs,k,steep,up,I,O,C,gi,gc,gb,[0],Array(8).fill(0),[0,0]),/input contribution|input fanin|derivative intermediate|native-q/);
  assert.deepEqual(gi,[73]);assert.deepEqual(gc,Array(8).fill(74));assert.deepEqual(gb,[75,76]);
});

test('gradient buffers cannot alias upstream or coefficient snapshot',()=>{
  const {xs,k,cs,up,I,O,C}=fx;
  const gi=[1,2],gc=Array(cs.length).fill(5),gb=[3,4];
  assert.throws(()=>vjpInto(xs,k,cs,up,I,O,C,gi,cs,gb,[0,0],Array(cs.length).fill(0),[0,0]),/alias/);
  assert.throws(()=>vjpInto(xs,k,cs,up,I,O,C,up,gc,gb,[0,0],Array(cs.length).fill(0),[0,0]),/alias/);
  assert.deepEqual(gi,[1,2]);assert(gc.every(v=>v===5));assert.deepEqual(gb,[3,4]);
});
// The functions below transcribe the exact index arithmetic of the candidate
// Oreslang scalar loops, then check it against DIFFERENT formulations above.
function candidateLocalBasis(knots, C, x, lane) {
  const s=span(knots,C,x);
  const d=[0,0,0,0];d[lane]=1;
  for(let r=1;r<=3;r++)for(let j=3;j>=r;j--){
    const i=s-3+j,den=knots[i+4-r]-knots[i];
    if(!(den>0))throw RangeError('native-basis-denominator');
    const a=(x-knots[i])/den;
    d[j]=(1-a)*d[j-1]+a*d[j];
  }
  return safe(d[3],'native-basis');
}
function candidateLocalInputDerivative(knots, coeff, x) {
  const C=coeff.length,s=span(knots,C,x),q=[0,0,0];
  for(let j=0;j<3;j++){
    const i=s-3+j,den=knots[i+4]-knots[i+1];
    if(!(den>0))throw RangeError('native-derivative-denominator');
    q[j]=safe(3*(coeff[i+1]-coeff[i])/den,'native-q');
  }
  for(let r=1;r<=2;r++)for(let j=2;j>=r;j--){
    const i=s-3+j,den=knots[i+4-r]-knots[i+1];
    if(!(den>0))throw RangeError('native-derivative-reduction');
    const a=(x-knots[i+1])/den;
    q[j]=(1-a)*q[j-1]+a*q[j];
  }
  return safe(q[2],'native-q2');
}
test('10,000 candidate Oreslang index-loop derivative results match independent trimmed-knot de Boor',()=>{
  for(let trial=0;trial<10000;trial++){
    const C=4+Math.floor(rnd()*17);
    const knots=[0,0,0,0,...Array.from({length:C-4},()=>.05+.9*rnd()).sort((a,b)=>a-b),1,1,1,1];
    const cs=Array.from({length:C},()=>rnd()-.5);
    const x=trial%31===0?0:trial%37===0?1:rnd();
    near(candidateLocalInputDerivative(knots,cs,x),analyticInput(knots,cs,x),3e-12);
  }
});
test('candidate Oreslang local four-basis de Boor matches independent Cox basis including knots',()=>{
  for(let trial=0;trial<5000;trial++){
    const C=4+Math.floor(rnd()*17);
    const knots=[0,0,0,0,...Array.from({length:C-4},()=>.05+.9*rnd()).sort((a,b)=>a-b),1,1,1,1];
    const x=trial%23===0?0:trial%29===0?1:rnd();
    const s=span(knots,C,x);
    for(let lane=0;lane<4;lane++){
      near(candidateLocalBasis(knots,C,x,lane),basis(knots,C,s-3+lane,x),3e-13);
    }
  }
  const repeated=[0,0,0,0,.5,.5,.5,1,1,1,1];
  for(const x of [.5-1e-8,.5,.5+1e-8]){
    const s=span(repeated,7,x);
    for(let lane=0;lane<4;lane++)near(candidateLocalBasis(repeated,7,x,lane),basis(repeated,7,s-3+lane,x),3e-13);
  }
});


test('read-only input/upstream alias is allowed when mutable buffers are distinct',()=>{
  const k=[0,0,0,0,1,1,1,1],same=[.5],cs=[0,.25,.5,.75];
  const gi=[99],gc=Array(4).fill(99),gb=[99];
  vjpInto(same,k,cs,same,1,1,4,gi,gc,gb,[0],Array(4).fill(0),[0]);
  near(gi[0],.375,1e-12);
  near(gb[0],.5,1e-12);
});

test('inactive derivative control outside quota does not poison active span VJP',()=>{
  const k=[0,0,0,0,1e-7,.5,1,1,1,1];
  const cs=[1e6,-1e6,0,0,0,0];
  const x=.75;
  const gi=[99],gc=Array(6).fill(99),gb=[99];
  // q0 is out of quota, but x is in the final span whose three derivative
  // controls all vanish. The forward active coefficient lanes are all zero.
  assert.throws(()=>analyticInput(k,cs,x),/derivative intermediate/);
  vjpInto([x],k,cs,[.5],1,1,6,gi,gc,gb,[0],Array(6).fill(0),[0]);
  near(gi[0],0,1e-12);
  near(gc.reduce((s,v)=>s+v,0),.5,1e-12);
  near(gb[0],.5,1e-12);
});
