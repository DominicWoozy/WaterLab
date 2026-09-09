import {
  common,
  NEIGHBOR_CACHE_BASE,
  recordNeighbor,
  cachedNeighbors as neighbors,
  neighbors as gridNeighbors,
} from './common.ts';
import { tensionCommon } from './surface-tension-shaders.ts';
const bindings = /* wgsl */ `
struct Reaction { linear:vec4f, angular:vec4f }
@group(0) @binding(1) var<storage,read> input:array<Particle>;
@group(0) @binding(2) var<storage,read_write> output:array<Particle>;
@group(0) @binding(3) var<storage,read> starts:array<u32>;
@group(0) @binding(4) var<storage,read> aux:array<vec4f>;
@group(0) @binding(5) var<storage,read_write> auxOut:array<vec4f>;
@group(0) @binding(6) var<storage,read> duck:Duck;
@group(0) @binding(7) var<storage,read_write> reactions:array<Reaction>;
@group(0) @binding(8) var<storage,read_write> surface:array<vec4f>;
`;
const kernel = (body: string, writeNeighborCache = false) =>
  common +
  tensionCommon +
  (writeNeighborCache
    ? bindings.replace(
        '@binding(3) var<storage,read>',
        '@binding(3) var<storage,read_write>',
      )
    : bindings) +
  /* wgsl */ `
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=P.counts.x){return;}${body}
}`;
export const fluidShaders: Record<string, string> = {
  initialize: kernel(/* wgsl */ `
 var nx=60u;var ny=5u;
 if(P.counts.y==30000u){nx=75u;ny=8u;}
 if(P.counts.y==50000u){nx=100u;ny=10u;}
 let p=vec3f(-1.7+f32(i%nx)*3.4/f32(nx-1u),-.917+f32(i/(nx*50u))*.36/f32(ny-1u),-1.2+f32((i/nx)%50u)*2.4/49.);
 output[i]=Particle(vec4f(p,1.),vec4f(p,1.),vec4f(0.,0.,0.,f32(i)));
 `),
  predict:
    common +
    bindings +
    /* wgsl */ `
 fn random(x:f32)->f32 {return fract(sin(x*127.1+P.clock.y*311.7)*43758.5453);}
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=P.counts.x){return;}var a=input[i];var p=a.pos.xyz;var v=a.vel.xyz;
 if(i>=P.counts.z){let f=f32(i);p=vec3f(P.motion.z+(random(f)-.5)*.29,1.8+random(f+33.)*.25,P.motion.w+(random(f+61.)-.5)*.29);v=vec3f(0.,-1.4,0.);a.vel.w=f;a.pos.w=1.;}
 a.old=vec4f(p,select(0.,1.,i<P.counts.z));
 let dt=P.clock.x;let t=P.clock.y;v.y-=P.clock.w*dt;
 v.x+=(P.forces.z+sin(t*2.1+p.z*3.)*P.forces.y*2.)*dt;v.z+=cos(t*1.7+p.x*2.)*P.forces.y*dt;
 let d=p-P.brush.xyz;let w=exp(-dot(d,d)/.32)*P.brush.w;
 v+=vec3f(P.motion.x-d.z*3.,.65,P.motion.y+d.x*3.)*w*dt*12.;
 let sd=p.xz-P.splash.xy;let sw=exp(-dot(sd,sd)/.18)*P.splash.z;
 v+=vec3f(sd.x*2.,1.6,sd.y*2.)*sw;a.pos=vec4f(bound(p+limited(v,12.)*dt),a.pos.w);output[i]=a;
 }`,
  lambda: kernel(
    /* wgsl */ `
 let p=input[i].pos.xyz;let wall=wallSupport(p)+duckSupport(p,duck);
 var rho=wall.w;var sum=0.;var grad=wall.xyz;var count=0u;
 // The following correction reads these same positions. Cache while gathering
 // pressure, then rebuild at the next lambda pass after positions have changed.
 ${gridNeighbors(`rho+=q*q;grad+=gradient;sum+=dot(gradient,gradient);${recordNeighbor()}`)}
 starts[${NEIGHBOR_CACHE_BASE}u+i]=count;
 // Neighbor constraints update simultaneously. Full Jacobi corrections overshoot
 // in the dense bulk and become velocity impulses on the next pass. Relax the
 // constraint multiplier (including boundary reactions), not the fluid velocity.
 auxOut[i]=vec4f(-.25*max(rho/REST-1.,0.)/(sum+dot(grad,grad)+2.),rho,0.,0.);
 `,
    true,
  ),
  correct: kernel(/* wgsl */ `
 let p=input[i].pos.xyz;let pressure=aux[i].x;var delta=-pressure*wallSupport(p).xyz;
 ${neighbors('delta-=(pressure+aux[j].x)*gradient;')}
 let boundaryDelta=-pressure*duckSupport(p,duck).xyz;delta+=boundaryDelta;
 let limiter=min(1.,.017*P.clock.z/max(length(delta),1e-8));var next=bound(p+delta*limiter);
 let hull=duckHull(next,duck);let contact=limited(hull.xyz*max(0.,.004-hull.w),.025*P.clock.z);next=bound(next+contact);
 let impulse=-particleMass()*(boundaryDelta*limiter+contact)/P.clock.x;
 var reaction=Reaction(vec4f(0.),vec4f(0.));if(P.counts.w==0u){reaction=reactions[i];}
 reaction.linear+=vec4f(impulse,0.);reaction.angular+=vec4f(cross(p-duck.pos.xyz,impulse),0.);reactions[i]=reaction;
 var a=input[i];a.pos=vec4f(next,a.pos.w);output[i]=a;
 `),
  velocity: kernel(/* wgsl */ `
 var a=input[i];var v=(a.pos.xyz-a.old.xyz)/P.clock.x;if(a.old.w<.5){v=vec3f(0.,-1.4,0.);}
 a.vel=vec4f(limited(v,12.)*pow(.998,P.clock.x*60.),a.vel.w);output[i]=a;
 `),
  viscosity: kernel(/* wgsl */ `
 let p=input[i].pos.xyz;let v=input[i].vel.xyz;var delta=vec3f(0.);var capillary=vec3f(0.);
 ${neighbors(`delta+=(input[j].vel.xyz-v)*q*q;
 if(P.forces.w>0.){
  let f=surfacePair(surface[i],surface[j],diff,r,aux[i].z,aux[j].z);let n=diff/r;
  // Pairwise radial dissipation resolves capillary oscillation at the fixed dt.
  // Equal/opposite and central; no damping of rigid translation or rotation.
  let damping=min(2.*sqrt(abs(f)/max(r,.15*h())),.25/(P.clock.x*max(1.,max(aux[i].z,aux[j].z))));
  capillary+=n*(damping*dot(input[j].vel.xyz-v,n)-select(0.,f,P.forces.w==1.));
 }`)}
 // Preserve the viscosity rate per simulated second when substepping.
 var a=input[i];a.vel=vec4f(limited(v+delta*(.002+P.forces.x*.065)*(P.clock.x*60.)+capillary*P.clock.x,12.),a.vel.w);output[i]=a;
 `),
  factor: kernel(/* wgsl */ `
 let p=input[i].pos.xyz;let wall=wallSupport(p)+duckSupport(p,duck);var rho=wall.w;var sum=0.;var nearby=0.;var grad=wall.xyz;
 ${gridNeighbors('rho+=q*q;grad+=gradient;sum+=dot(gradient,gradient);nearby+=1.;')}
 // Include boundary support in normals, but add no wall attraction.
 surface[i]=vec4f(limited(-h()*grad,2.),(rho+1.)/REST);
 var f=0.;if(nearby>=12.&&rho>REST*.4){f=1./max(sum+dot(grad,grad),1e-6);}
 auxOut[i]=vec4f(f,rho/REST,nearby,0.);
 `),
  residual: kernel(/* wgsl */ `
 let p=input[i].pos.xyz;let v=input[i].vel.xyz;
 var rate=-dot(v,wallSupport(p).xyz)-dot(v-duckVelocity(p,duck),duckSupport(p,duck).xyz);
 ${neighbors('rate+=dot(input[j].vel.xyz-v,gradient);')}
 auxOut[i]=vec4f(max(rate,0.)*aux[i].x,rate,aux[i].y,0.);
 `),
  project: kernel(/* wgsl */ `
 let p=input[i].pos.xyz;var v=input[i].vel.xyz;let pressure=aux[i].x;var delta=pressure*wallSupport(p).xyz;
 ${neighbors('delta+=(pressure+aux[j].x)*gradient;')}
 let boundaryDelta=.5*pressure*duckSupport(p,duck).xyz;let before=v+.5*delta;v=before+boundaryDelta;
 let hull=duckHull(p,duck);if(hull.w<.018){let relative=v-duckVelocity(p,duck);let normalPart=dot(relative,hull.xyz)*hull.xyz;
 v-=min(dot(relative,hull.xyz),0.)*hull.xyz;v-=(relative-normalPart)*(1.-pow(.975,P.clock.x*60.))*(1.-smoothstep(.004,.018,hull.w));}
 let limiter=min(1.,12./max(length(v),1e-8));let impulse=-(v-before)*limiter*particleMass();
 reactions[i].linear+=vec4f(impulse,0.);reactions[i].angular+=vec4f(cross(p-duck.pos.xyz,impulse),0.);
 v*=limiter;
 if(p.x<=-1.77999){v.x=max(v.x,0.);}if(p.x>=1.77999){v.x=min(v.x,0.);}
 if(p.y<=-.91699){v.y=max(v.y,0.);}if(p.y>=3.79999){v.y=min(v.y,0.);}
 if(p.z<=-1.27999){v.z=max(v.z,0.);}if(p.z>=1.27999){v.z=min(v.z,0.);}
 var a=input[i];a.vel=vec4f(v,a.vel.w);output[i]=a;
 `),
};
