// Independent, source-free B-spline arithmetic oracle. No private repo bytes.
// Evaluates one uniform cubic span through de Boor rather than copied basis code.
import test from 'node:test';
import assert from 'node:assert/strict';

const cubic = (t, c) => {
  if (t < 0 || t > 1 || !Number.isFinite(t) || c.length !== 4)
    throw RangeError('invalid cubic input');
  const knots = [0,1,2,3,4,5,6,7];
  const u = 3+t;
  const d = [...c];
  for (let r=1;r<=3;r++) for(let j=3;j>=r;j--) {
    const alpha=(u-knots[j])/(knots[j+4-r]-knots[j]);
    d[j]=(1-alpha)*d[j-1]+alpha*d[j];
  }
  return d[3];
};
const basis = t => [(1-t)**3/6,(3*t**3-6*t*t+4)/6,
                    (-3*t**3+3*t*t+3*t+1)/6,t**3/6];
const derivative = (t,c) => {
  const u=1-t;
  const db=[-u*u/2,(9*t*t-12*t)/6,(-9*t*t+6*t+3)/6,t*t/2];
  return db.reduce((s,v,i)=>s+v*c[i],0);
};
const check=(a,b,tol=1e-7)=>assert.ok(Math.abs(a-b)<=tol,String(a)+" != "+String(b));
const finiteDiff=(f,x,h=1e-6)=>(f(x+h)-f(x-h))/(2*h);
let seed=0x1234abcd;
const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};

test('de Boor and explicit basis match on 10,000 seeded coefficients/coordinates',()=>{
  for (let n=0;n<10000;n++) {
    const t=rand(),c=Array.from({length:4},()=>4*rand()-2);
    const explicit=basis(t).reduce((s,b,i)=>s+b*c[i],0);
    check(cubic(t,c),explicit,2e-14);
  }
});
test('analytical input and coefficient gradients match independent differences',()=>{
  for (let n=0;n<2000;n++) {
    const t=.05+.9*rand(),c=Array.from({length:4},()=>2*rand()-1);
    check(derivative(t,c),finiteDiff(v=>cubic(v,c),t),2e-9);
    for(let k=0;k<4;k++) {
      const p=c[k];
      const fd=finiteDiff(v=>cubic(t,c.map((z,j)=>j===k?v:z)),p);
      check(basis(t)[k],fd,2e-9);
    }
  }
});
test('MFKAN shared-low fan-in derivative agrees with finite differences',()=>{
  const lowCoeff=[.15,.25,.35,.45],highCoeff=[.1,.25,.5,.3],t=.37,mix=.4;
  const predict=c=>{
    const low=cubic(t,c);
    return low + mix*(.2*t+.4*low+.1)+(1-mix)*cubic(low,highCoeff);
  };
  const low=cubic(t,lowCoeff);
  const lowAdjoint=1+mix*.4+(1-mix)*derivative(low,highCoeff);
  for(let k=0;k<4;k++) {
    const analytic=lowAdjoint*basis(t)[k];
    check(analytic,finiteDiff(v=>predict(lowCoeff.map((z,j)=>j===k?v:z)),lowCoeff[k]),2e-9);
  }
});
test('bounded residual blend rejects overflow after valid component admission',()=>{
  const max=1e6;
  const checked=(lo,lin,nonlin,mix)=>{
    if (![lo,lin,nonlin,mix].every(Number.isFinite) ||
        [lo,lin,nonlin].some(x=>Math.abs(x)>max) || mix<0 || mix>1)
      throw RangeError('invalid input');
    const output=lo+mix*lin+(1-mix)*nonlin;
    if (!Number.isFinite(output)||Math.abs(output)>max)throw RangeError('invalid output');
    return output;
  };
  check(checked(.5,.4,.3,.25),.825,1e-14);
  assert.throws(()=>checked(1e6,1e6,1e6,.5),/invalid output/);
  assert.throws(()=>checked(NaN,1,1,.5),/invalid input/);
});
