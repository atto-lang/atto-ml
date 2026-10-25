// HOST-ONLY numeric oracle. Not imported by Oreslang guest code.
// Models one clamped/local uniform cubic B-spline span, t in [0,1].
// This is not a general multi-knot KAN evaluator and not a production trainer.
const MAX_ELEMENTS = 1_000_000;
const MAX_ABS = 1e6;
function requireFinite(x, label) {
  if (typeof x !== 'number' || !Number.isFinite(x) || Math.abs(x) > MAX_ABS)
    throw new RangeError(`${label}: expected bounded finite f64`);
}
function requireUnit(t, label) {
  requireFinite(t, label);
  if (t < 0 || t > 1) throw new RangeError(`${label}: t must be in [0,1]`);
}
function requireArray(a, expected, label) {
  if (!(Array.isArray(a) || a instanceof Float64Array) || a.length !== expected)
    throw new RangeError(`${label}: shape mismatch`);
  for (let i=0;i<a.length;i++) requireFinite(a[i], `${label}[${i}]`);
}
function checkedSize(a, b) {
  if (!Number.isSafeInteger(a) || !Number.isSafeInteger(b) || a < 1 || b < 1 || a > MAX_ELEMENTS || b > MAX_ELEMENTS || a > Math.floor(MAX_ELEMENTS/b))
    throw new RangeError('tensor shape exceeds bounded profile');
  return a*b;
}
export function cubicBasis(t) {
  requireUnit(t, 'spline coordinate');
  const u=1-t, t2=t*t;
  return [u*u*u/6, (3*t2*t-6*t2+4)/6,
          (-3*t2*t+3*t2+3*t+1)/6, t2*t/6];
}
export function cubicBasisDerivative(t) {
  requireUnit(t, 'spline coordinate');
  const u=1-t;
  return [-u*u/2, (9*t*t-12*t)/6,
          (-9*t*t+6*t+3)/6, t*t/2];
}
export function cubicEdge(t, coeff) {
  requireArray(coeff, 4, 'spline coefficients');
  const b=cubicBasis(t);
  const output=b.reduce((total, basis, i)=>total+basis*coeff[i],0);
  requireFinite(output,'spline output');
  return output;
}
export function cubicEdgeVjp(t, coeff, upstream=1) {
  requireArray(coeff,4,'spline coefficients');
  requireFinite(upstream,'upstream gradient');
  const b=cubicBasis(t), db=cubicBasisDerivative(t);
  const inputGrad=upstream*db.reduce((acc,d,i)=>acc+d*coeff[i],0);
  const coeffGrad=b.map(v=>upstream*v);
  requireFinite(inputGrad,'spline input gradient');
  coeffGrad.forEach((v,i)=>requireFinite(v,`spline coefficient gradient[${i}]`));
  return {inputGrad,coeffGrad};
}
export function validateLayer(layer, input) {
  if (!layer || typeof layer !== 'object') throw new TypeError('missing KAN layer');
  const {inputs,outputs,coeff,bias}=layer;
  const edgeCount=checkedSize(inputs,outputs);
  if (edgeCount > Math.floor(MAX_ELEMENTS/4)) throw new RangeError('coefficient quota exceeded');
  requireArray(input,inputs,'KAN input');
  for (let i=0;i<input.length;i++) requireUnit(input[i],`KAN input[${i}]`);
  requireArray(coeff,edgeCount*4,'KAN coefficients');
  requireArray(bias,outputs,'KAN bias');
  return edgeCount;
}
export function kanLayerForward(layer,input) {
  validateLayer(layer,input);
  const result = [];
  for(let o=0;o<layer.outputs;o++) {
    let value=layer.bias[o];
    for(let i=0;i<layer.inputs;i++) {
      const start=(o*layer.inputs+i)*4;
      const basis=cubicBasis(input[i]);
      for(let k=0;k<4;k++) value+=layer.coeff[start+k]*basis[k];
    }
    requireFinite(value,`KAN output[${o}]`);
    result.push(value);
  }
  return result;
}
export function kanLayerVjp(layer,input,upstream) {
  validateLayer(layer,input);
  requireArray(upstream,layer.outputs,'KAN upstream');
  const inputGrad=new Array(layer.inputs).fill(0);
  const coeffGrad=new Array(layer.coeff.length).fill(0);
  const biasGrad=[...upstream];
  for(let o=0;o<layer.outputs;o++) for(let i=0;i<layer.inputs;i++) {
    const offset=(o*layer.inputs+i)*4;
    const b=cubicBasis(input[i]), db=cubicBasisDerivative(input[i]);
    for(let k=0;k<4;k++) {
      coeffGrad[offset+k]+=upstream[o]*b[k];
      inputGrad[i]+=upstream[o]*layer.coeff[offset+k]*db[k];
    }
  }
  inputGrad.forEach((v,i)=>requireFinite(v,`KAN input gradient[${i}]`));
  coeffGrad.forEach((v,i)=>requireFinite(v,`KAN coefficient gradient[${i}]`));
  return {inputGrad,coeffGrad,biasGrad};
}
function requireMultifidelity(config,x) {
  if (!(Array.isArray(x) || x instanceof Float64Array) || x.length < 1)
    throw new RangeError('MFKAN input must be a nonempty numeric vector');
  if (!config || !config.low || !config.highNonlinear) throw new TypeError('missing MFKAN stages');
  if (config.low.inputs!==x.length || config.low.outputs!==1)
    throw new RangeError('low stage must produce one output');
  if (config.highNonlinear.inputs!==x.length+1 || config.highNonlinear.outputs!==1)
    throw new RangeError('high nonlinear KAN expects [x,lowPrediction]');
  requireArray(config.highLinear,x.length+2,'high affine parameters');
  requireUnit(config.mix,'MFKAN mixture');
  // The current local-span profile requires the *low prediction* to be
  // normalized into [0,1] by the caller's explicit data contract.
}
export function mfkanForward(config,x) {
  requireMultifidelity(config,x);
  const low=kanLayerForward(config.low,x)[0];
  requireUnit(low,'low prediction for high-stage spline');
  const features=[...x,low];
  const linear=config.highLinear[features.length] + features.reduce((s,v,i)=>s+v*config.highLinear[i],0);
  const nonlinear=kanLayerForward(config.highNonlinear,features)[0];
  const high=low + config.mix*linear + (1-config.mix)*nonlinear;
  requireFinite(linear,'linear correction');
  requireFinite(high,'high prediction');
  return {low,high,linear,nonlinear};
}
export function mfkanVjp(config,x,upstream=1,{freezeLow=true}={}) {
  requireFinite(upstream,'MFKAN upstream');
  const {low,linear,nonlinear}=mfkanForward(config,x);
  const features=[...x,low];
  const highVjp=kanLayerVjp(config.highNonlinear,features,[upstream*(1-config.mix)]);
  const highLinearGrad=features.map(f=>upstream*config.mix*f);
  highLinearGrad.push(upstream*config.mix); // bias
  const featureGradient=features.map((_,i)=>upstream*config.mix*config.highLinear[i]+highVjp.inputGrad[i]);
  const lowUpstream=upstream+featureGradient.at(-1); // skip + correction dependency
  const lowVjp=kanLayerVjp(config.low,x,[lowUpstream]);
  const gradients={
    inputGrad: x.map((_,i)=>featureGradient[i]+lowVjp.inputGrad[i]),
    highLinearGrad, highNonlinearCoeffGrad:highVjp.coeffGrad,
    highNonlinearBiasGrad:highVjp.biasGrad,
    lowCoeffGrad: freezeLow ? new Array(config.low.coeff.length).fill(0) : lowVjp.coeffGrad,
    lowBiasGrad: freezeLow ? [0] : lowVjp.biasGrad,
    mixGrad:upstream*(linear-nonlinear),
  };
  for (const [name,value] of Object.entries(gradients)) {
    if (Array.isArray(value)) value.forEach((v,i)=>requireFinite(v,`MFKAN ${name}[${i}]`));
    else requireFinite(value,`MFKAN ${name}`);
  }
  return gradients;
}
