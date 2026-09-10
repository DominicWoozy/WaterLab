// General free-surface cohesion. Inspired by Akinci et al. 2013, with our
// density-gradient normal estimate and central curvature projection. No sheet
// classifier, 2D area/thickness state, adhesion, or particle splitting.
export const tensionCommon = /* wgsl */ `
fn surfaceWeight(a:vec4f)->f32 {
 return max(1.-smoothstep(.7,1.05,a.w),smoothstep(.08,.35,length(a.xyz)));
}
fn surfacePair(a:vec4f,b:vec4f,diff:vec3f,r:f32,ni:f32,nj:f32)->f32 {
 return surfacePairWeighted(a,b,diff,r,ni,nj,max(surfaceWeight(a),surfaceWeight(b)));
}
fn surfacePairWeighted(a:vec4f,b:vec4f,diff:vec3f,r:f32,ni:f32,nj:f32,coverage:f32)->f32 {
 let x=r/h();let q=1.-x;let c=q*q*q*x*x*x;
 // Akinci's cohesion kernel: short-range repulsion, longer-range attraction.
 let cohesion=32./(3.14159265*h()*h()*h())*select(2.*c-1./64.,c,x>.5);
 let correction=2./max(.5,a.w+b.w);
 // Scene-calibrated coefficient, not a claim of SI-calibrated surface tension.
 let force=.04*correction*(1000.*particleMass()*cohesion+dot(a.xyz-b.xyz,diff/r))*coverage;
 let cap=.10/(P.clock.x*max(1.,max(ni,nj)));
 return clamp(force,-cap,cap);
}
`;
