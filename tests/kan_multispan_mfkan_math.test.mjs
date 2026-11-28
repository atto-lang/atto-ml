// Source-free independent Cox-de Boor MFKAN chain-rule oracle.
// Verifies mathematics only, never Oreslang compiler/backend execution.
import test from 'node:test';
import assert from 'node:assert/strict';

const near=(a,b,e=3e-7)=>assert.ok(Math.abs(a-b)<=e,String(a)+' vs '+String(b));
const diff=(f,x)=>(f(x+1e-6)-f(x-1e-6))/2e-6;
function basis(k,i,p,x) {
  if(!p)return k[i]<=x&&x<k[i+1]?1:0;
  const l=k[i+p]-k[i],r=k[i+p+1]-k[i+1];
  return (l>0?(x-k[i])/l*basis(k,i,p-1,x):0)
    +(r>0?(k[i+p+1]-x)/r*basis(k,i+1,p-1,x):0);
}
function dbasis(k,i,x) {
  const l=k[i+3]-k[i],r=k[i+4]-k[i+1];
  return (l>0?3*basis(k,i,2,x)/l:0)
       -(r>0?3*basis(k,i+1,2,x)/r:0);
}
function spline(k,c,x) {
  if(!Array.isArray(k)||!Array.isArray(c)||c.length<4||c.length>256||
      k.length!==c.length+4)throw RangeError('SHAPE');
  if(![...k,...c,x].every(v=>Number.isFinite(v)&&Math.abs(v)<=1e6))
    throw RangeError('FINITE');
  const n=c.length,a=k[3],b=k[n];
  if(!(a<b)||x<a||x>b||k.slice(0,4).some(v=>v!==a)||
      k.slice(n).some(v=>v!==b))throw RangeError('DOMAIN');
  let rep=1;
  for(let i=1;i<k.length;i++){
    if(k[i]<k[i-1])throw RangeError('ORDER');
    rep=k[i]===k[i-1]?rep+1:1;
    if(i>=4&&i<n&&rep>=4)throw RangeError('MULTIPLICITY');
  }
  for(let i=4;i<n;i++)if(k[i]<=a||k[i]>=b)throw RangeError('INTERIOR');
  if(x===b)return c.at(-1);
  return c.reduce((s,v,i)=>s+v*basis(k,i,3,x),0);
}
function dspline(k,c,x) {
  spline(k,c,x);
  const xx=x===k[c.length]?x-(k[c.length]-k[3])*1e-10:x;
  return c.reduce((s,v,i)=>s+v*dbasis(k,i,xx),0);
}
const bounded=x=>{if(!Number.isFinite(x)||Math.abs(x)>1e6)throw RangeError('BOUND');return x};
function forward(c,x) {
  [x,c.wx,c.wl,c.bias].forEach(bounded);
  if(!Number.isFinite(c.mix)||c.mix<0||c.mix>1)throw RangeError('MIX');
  const low=spline(c.lk,c.lc,x);
  const linear=bounded(c.wx*x+c.wl*low+c.bias);
  const nonlin=bounded(spline(c.fk,c.fc,x)+spline(c.pk,c.pc,low));
  return bounded(low+c.mix*linear+(1-c.mix)*nonlin);
}
function vjp(c,x,up,index,freeze=false) {
  bounded(up);forward(c,x);
  if(!Number.isInteger(index)||index<0||index>=c.lc.length)throw RangeError('LANE');
  const low=spline(c.lk,c.lc,x);
  const adj=bounded(1+c.mix*c.wl+(1-c.mix)*dspline(c.pk,c.pc,low));
  const input=bounded(up*(adj*dspline(c.lk,c.lc,x)
    +c.mix*c.wx+(1-c.mix)*dspline(c.fk,c.fc,x)));
  const lane=x===c.lk[c.lc.length]?(index===c.lc.length-1?1:0):basis(c.lk,index,3,x);
  return {input,lowCoeff:freeze?0:bounded(up*adj*lane)};
}
let seed=0x76ab53f2;
const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32};
const knots=()=>[0,0,0,0,...[.1+.8*rand(),.1+.8*rand()].sort((a,b)=>a-b),1,1,1,1];
const sample=()=>({lk:knots(),lc:Array.from({length:6},()=>.25+.5*rand()),
  fk:knots(),fc:Array.from({length:6},()=>rand()-.5),
  pk:knots(),pc:Array.from({length:6},()=>rand()-.5),
  wx:.1+.2*rand(),wl:.1+.3*rand(),bias:.05,mix:.2+.6*rand()});

test('source fixture: forward 0.825, input VJP 0.875, low coefficient VJP 0.55',()=>{
  const k=[0,0,0,0,.5,1,1,1,1];
  const c={lk:k,lc:[0,.25,.5,.75,1],fk:k,fc:Array(5).fill(.2),
    pk:k,pc:Array(5).fill(.1),wx:.2,wl:.4,bias:.1,mix:.25};
  near(forward(c,.5),.825,1e-14);
  near(vjp(c,.5,1,2).input,.875,2e-14);
  near(vjp(c,.5,1,2).lowCoeff,.55,2e-14);
  near(vjp(c,.5,1,2,true).lowCoeff,0);
});
test('1,500 seeded forward examples remain finite',()=>{
  for(let j=0;j<1500;j++)assert.ok(Number.isFinite(forward(sample(),.02+.96*rand())));
});
test('450 numerical input VJPs across two coupled low paths',()=>{
  for(let j=0;j<450;j++){
    const c=sample(),x=.03+.94*rand(),up=.2+.8*rand();
    if([...c.lk,...c.fk].some(k=>Math.abs(k-x)<3e-4))continue;
    near(vjp(c,x,up,2).input,up*diff(z=>forward(c,z),x));
  }
});
test('350 sweeps of every low-coefficient gradient lane',()=>{
  for(let j=0;j<350;j++){
    const c=sample(),x=.03+.94*rand(),up=.1+.9*rand();
    for(let i=0;i<c.lc.length;i++){
      const fd=up*diff(z=>forward({...c,lc:c.lc.map((v,k)=>k===i?z:v)},x),c.lc[i]);
      near(vjp(c,x,up,i).lowCoeff,fd);
    }
  }
});
test('freeze-low never suppresses input VJP',()=>{
  for(let j=0;j<150;j++){
    const c=sample(),x=.02+.96*rand(),i=Math.floor(rand()*c.lc.length);
    near(vjp(c,x,1,i).input,vjp(c,x,1,i,true).input,1e-14);
    assert.equal(vjp(c,x,1,i,true).lowCoeff,0);
  }
});
test('invalid mix, mismatched prediction domain and frozen invalid lanes reject',()=>{
  const c=sample();
  assert.throws(()=>forward({...c,mix:NaN},.35),/MIX/);
  assert.throws(()=>forward({...c,mix:-.1},.35),/MIX/);
  assert.throws(()=>forward({...c,pk:[0,0,0,0,.1,.15,.2,.2,.2,.2]},.35),/DOMAIN/);
  assert.throws(()=>vjp(c,.35,1,-1,true),/LANE/);
  assert.throws(()=>vjp(c,.35,1,c.lc.length,true),/LANE/);
  assert.throws(()=>vjp(c,.35,NaN,0,true),/BOUND/);
});
