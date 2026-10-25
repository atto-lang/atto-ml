// Independent Cox-de Boor HOST oracle; not actual Oreslang guest execution.
import test from 'node:test';
import assert from 'node:assert/strict';
const LIMIT=1e6;
function B(k,i,p,x){
 if(p===0)return k[i]<=x&&x<k[i+1]?1:0;
 const a=k[i+p]-k[i],b=k[i+p+1]-k[i+1];
 return (a>0?(x-k[i])/a*B(k,i,p-1,x):0)
      +(b>0?(k[i+p+1]-x)/b*B(k,i+1,p-1,x):0);
}
function dB(k,i,p,x){
 const a=k[i+p]-k[i],b=k[i+p+1]-k[i+1];
 return (a>0?p/a*B(k,i,p-1,x):0)-(b>0?p/b*B(k,i+1,p-1,x):0);
}
function predict(xs,k,c,b,I,O,C){
 return Array.from({length:O},(_,o)=>{
  let v=b[o];
  for(let i=0;i<I;i++)for(let j=0;j<C;j++){
   const basis=xs[i]===k[C]?Number(j===C-1):B(k,j,3,xs[i]);
   v+=c[(o*I+i)*C+j]*basis;
  }
  return v;
 });
}
function workspace(I,O,C){return {gx:Array(I).fill(77),gc:Array(I*O*C).fill(77),gb:Array(O).fill(77),
 wx:Array(I).fill(0),wc:Array(I*O*C).fill(0),wb:Array(O).fill(0),py:Array(O).fill(0),ps:Array(O).fill(0)}}
function gradients(xs,k,c,b,up,I,O,C,w){
 if(I<1||I>32||O<1||O>32||C<4||C>256||I*O*C>65536)throw RangeError('QUOTA');
 const reads=[xs,k,c,b,up],outs=[w.gx,w.gc,w.gb,w.wx,w.wc,w.wb,w.py,w.ps];
 if([...reads,...outs].some(v=>!Array.isArray(v)))throw TypeError('ARRAY');
 for(let z=0;z<outs.length;z++)
  if(reads.some(r=>r===outs[z])||outs.slice(z+1).includes(outs[z]))throw RangeError('ALIAS');
 if(xs.length!==I||k.length!==C+4||c.length!==I*O*C||b.length!==O||up.length!==O||
   w.gx.length!==I||w.wx.length!==I||w.gc.length!==I*O*C||w.wc.length!==I*O*C||
   w.gb.length!==O||w.wb.length!==O||w.py.length!==O||w.ps.length!==O)throw RangeError('SHAPE');
 if([...xs,...k,...c,...b,...up].some(v=>!Number.isFinite(v)||Math.abs(v)>LIMIT))throw RangeError('NUMERIC');
 const lo=k[3],hi=k[C];if(!(lo<hi)||k.slice(0,4).some(v=>v!==lo)||k.slice(C).some(v=>v!==hi))throw RangeError('KNOT');
 let repeated=1;
 for(let j=1;j<k.length;j++){
  if(k[j]<k[j-1])throw RangeError('ORDER');
  repeated=k[j]===k[j-1]?repeated+1:1;
  if(j>=4&&j<C&&repeated>=4)throw RangeError('MULTIPLICITY');
 }
 if(xs.some(x=>x<lo||x>hi)||k.slice(4,C).some(v=>v<=lo||v>=hi))throw RangeError('DOMAIN');
 const p=predict(xs,k,c,b,I,O,C);
 if(p.some(v=>!Number.isFinite(v)||Math.abs(v)>LIMIT))throw RangeError('FORWARD');
 w.py.splice(0,O,...p);
 w.wx.fill(0);w.wc.fill(0);
 for(let o=0;o<O;o++){
  w.wb[o]=up[o];
  for(let i=0;i<I;i++){
   const x=xs[i],refx=x===hi?hi-1e-12*(hi-lo):x,off=(o*I+i)*C;
   let derivative=0;
   for(let j=0;j<C;j++)derivative+=c[off+j]*dB(k,j,3,refx);
   if(!Number.isFinite(derivative)||Math.abs(derivative)>LIMIT)throw RangeError('DERIV');
   const product=up[o]*derivative;
   if(!Number.isFinite(product)||Math.abs(product)>LIMIT)throw RangeError('PRODUCT');
   w.wx[i]+=product;
   if(!Number.isFinite(w.wx[i])||Math.abs(w.wx[i])>LIMIT)throw RangeError('FANIN');
   for(let j=0;j<C;j++){
    const v=up[o]*(x===hi?Number(j===C-1):B(k,j,3,x));
    if(!Number.isFinite(v)||Math.abs(v)>LIMIT)throw RangeError('COEFF');
    w.wc[off+j]=v;
   }
  }
 }
 w.gx.splice(0,I,...w.wx);w.gc.splice(0,I*O*C,...w.wc);w.gb.splice(0,O,...w.wb);
}
const near=(a,b,tol=2e-7)=>assert.ok(Math.abs(a-b)<=tol+2e-6*Math.abs(b),String(a)+' vs '+String(b));
let seed=0x63417;const rnd=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32};
test('2x2 packed coefficient lanes, input fan-in, and bias gradients',()=>{
 const k=[0,0,0,0,1,1,1,1],c=[...Array(4).fill(.2),...Array(4).fill(.3),...Array(4).fill(.4),...Array(4).fill(.5)],w=workspace(2,2,4);
 gradients([.25,.75],k,c,[.1,-.1],[2,-3],2,2,4,w);
 near(w.gx[0],0);near(w.gx[1],0);near(w.gb[0],2);near(w.gb[1],-3);
 near(w.gc[0],.84375);near(w.gc[8],-1.265625);
});
test('1200 seeded random finite difference VJPs for inputs and coefficients',()=>{
 for(let t=0;t<1200;t++){
  const I=1+Math.floor(3*rnd()),O=1+Math.floor(3*rnd()),C=4+Math.floor(6*rnd());
  const ks=Array.from({length:C-4},()=>.03+.94*rnd()).sort((a,b)=>a-b);
  const k=[0,0,0,0,...ks,1,1,1,1],xs=Array.from({length:I},()=>.08+.84*rnd());
  if(xs.some(x=>k.some(z=>Math.abs(x-z)<.004)))continue;
  const c=Array.from({length:I*O*C},()=>2*rnd()-1),b=Array.from({length:O},()=>2*rnd()-1),up=Array.from({length:O},()=>2*rnd()-1),w=workspace(I,O,C);
  gradients(xs,k,c,b,up,I,O,C,w);
  const scalar=(a,d)=>predict(a,k,d,b,I,O,C).reduce((s,v,j)=>s+up[j]*v,0),h=1e-6;
  for(let i=0;i<I;i++){
   const a=xs.slice(),z=xs.slice();a[i]+=h;z[i]-=h;
   near(w.gx[i],(scalar(a,c)-scalar(z,c))/(2*h));
  }
  for(let tries=0;tries<2;tries++){
   const j=Math.floor(rnd()*c.length),a=c.slice(),z=c.slice();a[j]+=h;z[j]-=h;
   near(w.gc[j],(scalar(xs,a)-scalar(xs,z))/(2*h),6e-7);
  }
 }
});
test('right-hand input gradient at triple interior knot',()=>{
 const k=[0,0,0,0,.5,.5,.5,1,1,1,1],w=workspace(1,1,7);
 gradients([.5],k,[0,.1,.2,.3,.8,.5,.6],[0],[2],1,1,7,w);
 near(w.gx[0],6,1e-10);
});
test('zero upstream returns zero input, bias and coefficient VJPs',()=>{
 const w=workspace(1,1,4);gradients([.5],[0,0,0,0,1,1,1,1],[0,1,2,3],[0],[0],1,1,4,w);
 assert.ok([...w.gx,...w.gc,...w.gb].every(v=>Math.abs(v)<1e-10));
});
test('input gradient overflow preserves all three published buffers',()=>{
 const w=workspace(1,1,4);
 assert.throws(()=>gradients([.5],[0,0,0,0,1,1,1,1],[0,1,2,3],[0],[1e6],1,1,4,w),/PRODUCT/);
 assert.deepEqual(w.gx,[77]);assert.deepEqual(w.gc,[77,77,77,77]);assert.deepEqual(w.gb,[77]);
});
test('fan-in overflow across distinct outputs preserves destination arrays',()=>{
 const w=workspace(1,2,4);
 assert.throws(()=>gradients([.5],[0,0,0,0,1,1,1,1],[0,1,2,3,0,1,2,3],[0,0],[300000,300000],1,2,4,w),/FANIN/);
 assert.ok([...w.gx,...w.gc,...w.gb].every(v=>v===77));
});
test('bad buffer shape, upstream, knots, and output aliases fail closed',()=>{
 const k=[0,0,0,0,1,1,1,1],c=[0,1,2,3],w=workspace(1,1,4);
 assert.throws(()=>gradients([.5],k,c,[0],[Infinity],1,1,4,w));
 assert.throws(()=>gradients([.5],k,c.slice(1),[0],[1],1,1,4,w),/SHAPE/);
 assert.throws(()=>gradients([.5],k,c,[0],[1],1,1,4,{...w,wx:w.gx}),/ALIAS/);
 assert.ok([...w.gx,...w.gc,...w.gb].every(v=>v===77));
});
test('host-only 20-step SGD on nonlinear cubic basis reduces squared loss',()=>{
 const k=[0,0,0,0,1,1,1,1],x=[.5],c=[0,0,0,0],b=[0],target=1;
 const loss=()=>Math.pow(predict(x,k,c,b,1,1,4)[0]-target,2),initial=loss();
 for(let t=0;t<20;t++){
  const y=predict(x,k,c,b,1,1,4)[0],w=workspace(1,1,4);
  gradients(x,k,c,b,[2*(y-target)],1,1,4,w);
  for(let i=0;i<4;i++)c[i]-=.1*w.gc[i];
  b[0]-=.1*w.gb[0];
 }
 assert.ok(loss()<initial*.0001);
});
