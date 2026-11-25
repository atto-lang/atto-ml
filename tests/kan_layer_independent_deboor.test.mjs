// HOST-ONLY independent property tests: iterative de Boor vs recursive Cox.
import test from 'node:test';
import assert from 'node:assert/strict';

const basis=(k,i,p,x)=>{
  if(p===0)return k[i]<=x&&x<k[i+1]?1:0;
  const a=k[i+p]-k[i],b=k[i+p+1]-k[i+1];
  return (a>0?(x-k[i])/a*basis(k,i,p-1,x):0)+
         (b>0?(k[i+p+1]-x)/b*basis(k,i+1,p-1,x):0);
};
function cox(k,c,x) {
  if(x===k[c.length])return c.at(-1);
  return c.reduce((s,v,i)=>s+v*basis(k,i,3,x),0);
}
function deBoor(k,c,x) {
  let s=c.length-1;
  for(let i=3;i<c.length;i++)if(x>=k[i]&&x<k[i+1]){s=i;break}
  const d=c.slice(s-3,s+1);
  for(let r=1;r<=3;r++)for(let j=3;j>=r;j--){
    const i=s-3+j,den=k[i+4-r]-k[i];
    if(!(den>0))throw RangeError('degenerate');
    const a=(x-k[i])/den;
    d[j]=(1-a)*d[j-1]+a*d[j];
  }
  return d[3];
}
let seed=0x25310517;
const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32};
const near=(a,b,eps=4e-13)=>assert.ok(Math.abs(a-b)<=eps,String(a)+' != '+String(b));

test('12,000 independent cubic evaluations across clamped multi-span grids',()=>{
  for(let t=0;t<12000;t++){
    const n=4+Math.floor(rand()*15);
    const interior=Array.from({length:n-4},()=>.01+.98*rand()).sort((a,b)=>a-b);
    const k=[0,0,0,0,...interior,1,1,1,1];
    const c=Array.from({length:n},()=>2*rand()-1);
    const x=t%23===0?0:t%29===0?1:rand();
    near(deBoor(k,c,x),cox(k,c,x));
  }
});
test('independent packed 2-input, 2-output layer across 2,000 trials',()=>{
  for(let trial=0;trial<2000;trial++){
    const I=1+Math.floor(rand()*5),O=1+Math.floor(rand()*5);
    const n=4+Math.floor(rand()*12);
    const interior=Array.from({length:n-4},()=>.01+.98*rand()).sort((a,b)=>a-b);
    const k=[0,0,0,0,...interior,1,1,1,1];
    const xs=Array.from({length:I},()=>rand());
    const coeff=Array.from({length:I*O*n},()=>2*rand()-1);
    const bias=Array.from({length:O},()=>2*rand()-1);
    for(let o=0;o<O;o++){
      let got=bias[o],expected=bias[o];
      for(let i=0;i<I;i++){
        const start=(o*I+i)*n,c=coeff.slice(start,start+n);
        got+=deBoor(k,c,xs[i]);
        expected+=cox(k,c,xs[i]);
      }
      near(got,expected,1e-12);
    }
  }
});
test('triple interior knot and both endpoints retain exact branch policy',()=>{
  const k=[0,0,0,0,.5,.5,.5,1,1,1,1],c=[0,.1,.2,.3,.8,.5,.6];
  for(const x of [0,1,.5-1e-10,.5,.5+1e-10]){
    near(deBoor(k,c,x),cox(k,c,x),3e-12);
  }
});
test('full 65,536 packed-coefficient budget sums 32x32 constant edges',()=>{
  const I=32,O=32,n=64;
  const k=[0,0,0,0,...Array.from({length:n-4},(_,i)=>(i+1)/(n-3)),1,1,1,1];
  const c=Array(n).fill(.01),coeffCount=I*O*n;
  assert.equal(coeffCount,65536);
  for(let o=0;o<O;o++){
    let value=0;
    for(let i=0;i<I;i++)value+=deBoor(k,c,.5);
    near(value,.32,3e-12);
  }
});
