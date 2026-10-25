// Source-free independent numerical acceptance oracle for a proposed Oreslang
// clamped multi-span cubic spline. Not a guest compiler execution.
import test from 'node:test';
import assert from 'node:assert/strict';

const LIMIT=1e6;
function admit(knots,coeff,x) {
  if (!Array.isArray(knots)||!Array.isArray(coeff)||!Number.isInteger(coeff.length)||
      coeff.length<4||coeff.length>256||knots.length!==coeff.length+4)
    throw new RangeError('SHAPE');
  if(![...knots,...coeff,x].every(v=>typeof v==='number'&&Number.isFinite(v)&&Math.abs(v)<=LIMIT))
    throw new RangeError('NONFINITE_OR_RANGE');
  const n=coeff.length,lo=knots[3],hi=knots[n];
  if(!(lo<hi)||knots.slice(0,4).some(k=>k!==lo)||knots.slice(n).some(k=>k!==hi))
    throw new RangeError('CLAMPED');
  let repeat=1;
  for(let i=1;i<knots.length;i++) {
    if(knots[i]<knots[i-1])throw new RangeError('UNSORTED');
    repeat=knots[i]===knots[i-1]?repeat+1:1;
    if(i>=4&&i<n&&repeat>=4)throw new RangeError('MULTIPLICITY');
  }
  for(let i=4;i<n;i++)if(knots[i]<=lo||knots[i]>=hi)throw new RangeError('INTERIOR');
  if(x<lo||x>hi)throw new RangeError('DOMAIN');
  return n;
}
function span(knots,n,x) {
  for(let i=3;i<n;i++)if(x>=knots[i]&&x<knots[i+1])return i;
  return n-1;
}
function deboor(knots,coeff,x) {
  const n=admit(knots,coeff,x),s=span(knots,n,x);
  const d=coeff.slice(s-3,s+1);
  for(let r=1;r<=3;r++)for(let j=3;j>=r;j--){
    const i=s-3+j,den=knots[i+4-r]-knots[i];
    if(den<=0)throw new RangeError('DEGENERATE');
    const a=(x-knots[i])/den;
    d[j]=(1-a)*d[j-1]+a*d[j];
  }
  if(!Number.isFinite(d[3])||Math.abs(d[3])>LIMIT)throw new RangeError('OUTPUT_RANGE');
  return d[3];
}
function derivative(knots,coeff,x) {
  const n=admit(knots,coeff,x),s=span(knots,n,x);
  const q=[];
  for(let j=0;j<3;j++) {
    const i=s-3+j,den=knots[i+4]-knots[i+1];
    if(den<=0)throw new RangeError('DEGENERATE');
    const v=3*(coeff[i+1]-coeff[i])/den;
    if(!Number.isFinite(v)||Math.abs(v)>LIMIT)throw new RangeError('GRADIENT_RANGE');
    q.push(v);
  }
  for(let r=1;r<=2;r++)for(let j=2;j>=r;j--){
    const i=s-3+j,den=knots[i+4-r]-knots[i+1];
    if(den<=0)throw new RangeError('DEGENERATE');
    const a=(x-knots[i+1])/den;
    q[j]=(1-a)*q[j-1]+a*q[j];
  }
  if(!Number.isFinite(q[2])||Math.abs(q[2])>LIMIT)throw new RangeError('GRADIENT_RANGE');
  return q[2];
}
function dcoeff(knots,coeff,x,index,upstream) {
  const n=admit(knots,coeff,x);
  if(!Number.isInteger(index)||index<0||index>=n||
     !Number.isFinite(upstream)||Math.abs(upstream)>LIMIT)throw new RangeError('VJP');
  const s=span(knots,n,x);
  if(index<s-3||index>s)return 0;
  const local=Array(4).fill(0);local[index-(s-3)]=1;
  const d=[...local];
  for(let r=1;r<=3;r++)for(let j=3;j>=r;j--){
    const i=s-3+j,a=(x-knots[i])/(knots[i+4-r]-knots[i]);
    d[j]=(1-a)*d[j-1]+a*d[j];
  }
  const g=d[3]*upstream;
  if(!Number.isFinite(g)||Math.abs(g)>LIMIT)throw new RangeError('GRADIENT_RANGE');
  return g;
}
// Mathematically independent Cox-de Boor recursion (not the local worklist).
function cox(knots,i,p,x) {
  if(p===0)return knots[i]<=x && x<knots[i+1] ? 1 : 0;
  const ld=knots[i+p]-knots[i],rd=knots[i+p+1]-knots[i+1];
  return (ld>0?(x-knots[i])/ld*cox(knots,i,p-1,x):0) +
         (rd>0?(knots[i+p+1]-x)/rd*cox(knots,i+1,p-1,x):0);
}
function recursiveReference(knots,coeff,x) {
  const n=admit(knots,coeff,x);
  if(x===knots[n])return coeff[n-1];
  return coeff.reduce((sum,c,i)=>sum+c*cox(knots,i,3,x),0);
}
let seed=0x75aac131;
const rnd=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32};
function fixture() {
  const n=5+Math.floor(rnd()*7);
  const interior=Array.from({length:n-4},()=>.1+.8*rnd()).sort((a,b)=>a-b);
  const knots=[0,0,0,0,...interior,1,1,1,1];
  const coeff=Array.from({length:n},()=>rnd()*2-1);
  return {knots,coeff};
}
const near=(a,b,tol=1e-8)=>assert.ok(Math.abs(a-b)<=tol,String(a)+' vs '+String(b));
const central=(f,x,h=1e-6)=>(f(x+h)-f(x-h))/(2*h);

test('10,000 random multi-span evaluations agree with independent Cox-de Boor',()=>{
  for(let j=0;j<10000;j++){
    const {knots,coeff}=fixture(),x=rnd();
    near(deboor(knots,coeff,x),recursiveReference(knots,coeff,x),2e-13);
  }
});
test('cubic input derivative matches independent numerical difference across spans',()=>{
  for(let j=0;j<2000;j++){
    const {knots,coeff}=fixture(),x=.01+.98*rnd();
    if(knots.some(k=>Math.abs(x-k)<2e-4))continue;
    near(derivative(knots,coeff,x),central(v=>recursiveReference(knots,coeff,v),x),1e-7);
  }
});
test('per-coefficient VJP matches finite differences including inactive coefficients',()=>{
  for(let j=0;j<800;j++){
    const {knots,coeff}=fixture(),x=.02+.96*rnd(),up=2*rnd()-1;
    for(let i=0;i<coeff.length;i++){
      const direct=dcoeff(knots,coeff,x,i,up);
      const numeric=up*central(v=>recursiveReference(knots,coeff.map((z,k)=>k===i?v:z),x),coeff[i]);
      near(direct,numeric,2e-8);
    }
  }
});
test('closed endpoint convention evaluates to first/last coefficient',()=>{
  for(let j=0;j<200;j++){
    const {knots,coeff}=fixture();
    near(deboor(knots,coeff,knots[3]),coeff[0],1e-14);
    near(deboor(knots,coeff,knots[coeff.length]),coeff.at(-1),1e-14);
    near(dcoeff(knots,coeff,knots[coeff.length],coeff.length-1,1),1,1e-14);
    near(dcoeff(knots,coeff,knots[3],0,1),1,1e-14);
  }
});
test('multiple interior knot multiplicity up to three is admitted',()=>{
  const k=[0,0,0,0,.5,.5,.5,1,1,1,1],c=[0,.1,.2,.3,.4,.5,.6];
  for(const x of [.1,.49,.5,.51,.9])near(deboor(k,c,x),recursiveReference(k,c,x),1e-13);
});
test('partition of unity for multi-span local basis',()=>{
  for(let j=0;j<200;j++){
    const {knots,coeff}=fixture(),x=rnd();
    const total=coeff.reduce((a,_,i)=>a+dcoeff(knots,coeff,x,i,1),0);
    near(total,1,2e-13);
  }
});
test('reject malformed shapes, domain, invalid numbers, and knot order',()=>{
  const k=[0,0,0,0,.5,1,1,1,1],c=[0,.1,.2,.3,.4];
  const invalid=[
    [k.slice(0,-1),c,.4],[k,c.slice(1),.4],
    [[0,0,0,0,.9,.3,1,1,1],c,.4],
    [[0,0,0,0,.5,.5,.5,.5,1,1,1,1],[...c,.4,.5,.6],.5],
    [[0,0,0,0,.5,1,1,1,1],c,1.1],
    [[0,0,0,0,.5,1,1,1,1],[0,.1,.2,.3,NaN],.5],
    [[0,0,0,0,.5,1,1,1,1],c,NaN],
    [[0,0,0,0,.5,1,1,1,1],c,Infinity],
    [[0,0,0,0,.5,1,1,1,1],c,-.0001],
  ];
  for(const [ks,cs,x] of invalid)assert.throws(()=>deboor(ks,cs,x),RangeError);
  assert.throws(()=>deboor(Array(265).fill(0),Array(261).fill(0),0),/SHAPE/);
});
test('admitted inputs may still have a rejected out-of-range derivative',()=>{
  const knots=[0,0,0,0,.0000001,1,1,1,1],coeff=[1e6,-1e6,1e6,-1e6,1e6];
  assert.throws(()=>derivative(knots,coeff,0),/GRADIENT_RANGE/);
});
test('linear-coefficient shape has closed end-to-end monotone output',()=>{
  const knots=[0,0,0,0,.5,1,1,1,1],coeff=[0,.25,.5,.75,1];
  near(deboor(knots,coeff,0),0);
  near(deboor(knots,coeff,.5),.5);
  near(deboor(knots,coeff,1),1);
});

test('triple interior knot chooses right-hand derivative, distinct from left',()=>{
  const knots=[0,0,0,0,.5,.5,.5,1,1,1,1];
  const coeff=[0,.1,.2,.3,.8,.5,.6];
  const x=.5,h=1e-7;
  const right=(deboor(knots,coeff,x+h)-deboor(knots,coeff,x))/h;
  const left=(deboor(knots,coeff,x)-deboor(knots,coeff,x-h))/h;
  near(derivative(knots,coeff,x),right,2e-5);
  assert.ok(Math.abs(right-left)>.1);
});
