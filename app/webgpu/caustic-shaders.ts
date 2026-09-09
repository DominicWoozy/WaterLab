import { renderScene } from './render-shaders.ts';
import { LIGHT_FOOTPRINT, receiverScene } from './light-space.ts';

export const CAUSTIC_GRID = [384, 320] as const;
export const CAUSTIC_MAP = [1280, 1024] as const;
const grid = /* wgsl */ `
const NX=${CAUSTIC_GRID[0]}u;const NZ=${CAUSTIC_GRID[1]}u;
const SEED_SIZE=vec2f(${LIGHT_FOOTPRINT[0]}.,${LIGHT_FOOTPRINT[1]}.);
struct Photon {entry:vec4f,hit:vec4f,energy:vec4f,footprint:vec4f}
`;

// Forward light transport through up to eight air/water interfaces. Only the
// transmitted Fresnel branch is followed, except for total internal reflection.
export function causticTraceShader(filterable: boolean) {
  return (
    renderScene(filterable) +
    receiverScene +
    grid +
    /* wgsl */ `
@group(0) @binding(12) var<storage,read_write> photons:array<Photon>;
@group(0) @binding(16) var<uniform> lightControl:vec4u;
@group(0) @binding(17) var<storage,read_write> duckPhotons:array<u32>;
@group(0) @binding(18) var<storage,read_write> duckDraw:array<atomic<u32>>;
fn lightDensity(p:vec3f)->f32 {
 // The visual density sampler clips at -.96. Extend that final centimeter to
 // the solid floor, instead of inventing an air interface inside the platform.
 return density(vec3f(p.x,max(p.y,-.95),p.z));
}
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=NX||id.y>=NZ){return;}let index=id.x+NX*id.y;
 photons[index]=Photon(vec4f(0.),vec4f(0.),vec4f(0.),vec4f(0.));
 volumeTop=bitcast<f32>(bounds[0])-2.+.36;
 let sun=normalize(vec3f(-.6,1.,.35));var rd=-sun;
 let seed=(vec2f(id.xy)/vec2f(f32(NX-1u),f32(NZ-1u))-.5)*SEED_SIZE;
 let startY=max(max(volumeTop+.04,duck.pos.y+1.),1.);
 var ro=vec3f(seed.x,-.97,seed.y)+sun*((startY+.97)/sun.y);
 var first=vec3f(seed.x,-.97,seed.y);var energy=vec3f(1.);
 var inside=false;var interfaces=0u;var samples=0u;var internalReflections=0u;
 let traceWater=lightControl.x>0u&&S.light.z>.5;
 for(var segment=0;segment<9;segment++){
  let receiver=lightReceiver(ro,rd);
  let opaque=duckTrace(ro,rd,receiver.x);
  let end=min(receiver.x,opaque.x);
  var found=false;var near=0.;var far=0.;
  if(traceWater){
   let interval=boxHit(ro,rd);var t=max(0.,interval.x);let stop=min(end,interval.y);
   near=t;
   for(var step=0;step<768;step++){
    if(t>stop){break;}if(samples>=768u){return;}samples++;
    let d=lightDensity(ro+rd*t);
    if((d>S.config.x)!=inside){far=t;found=true;break;}
    if(t==stop){break;}near=t;t=min(stop,t+select(.026,.014,inside||d>.12));
   }
  }
  if(!found){
   if(opaque.y>=0.){
    if(inside){energy*=exp(-vec3f(1.25,.2,.065)*opaque.x);}
    let tri=u32(opaque.y)*6u;let a=triangles[tri];let b=triangles[tri+1u];let c=triangles[tri+2u];
    let n=qrotate(duck.rotation,normalize(cross(b.xyz-a.xyz,c.xyz-a.xyz)));
    let incidence=abs(dot(n,-rd));let radius=.04/sqrt(max(incidence,.1));
    photons[index]=Photon(vec4f(first,f32(interfaces)),vec4f(ro+rd*opaque.x,3.),vec4f(energy,f32(interfaces)),vec4f(n,radius));
    let slot=atomicAdd(&duckDraw[1],1u);duckPhotons[slot]=index;return;
   }
   if(receiver.y<.5){return;}
   if(inside){energy*=exp(-vec3f(1.25,.2,.065)*receiver.x);}
   let hit=ro+rd*receiver.x;
   photons[index]=Photon(vec4f(first,f32(interfaces)),vec4f(hit,receiver.y),vec4f(energy,f32(interfaces)),vec4f(0.,0.,0.,f32(internalReflections)));
   return;
  }
  if(interfaces>=8u){return;}
  for(var refine=0;refine<6;refine++){
   let mid=(near+far)*.5;
   if((lightDensity(ro+rd*mid)>S.config.x)!=inside){far=mid;}else{near=mid;}
  }
  let p=ro+rd*far;if(interfaces==0u){first=p;}
  if(inside){energy*=exp(-vec3f(1.25,.2,.065)*far);}
  var normal=normalAt(p);if(dot(normal,rd)>0.){normal=-normal;}
  let eta=select(1./1.333,1.333,inside);let transmitted=refract(rd,normal,eta);
  if(dot(transmitted,transmitted)<1e-8){rd=reflect(rd,normal);internalReflections++;}
  else{
   let cosine=max(0.,dot(-rd,normal));
   let transmittedCosine=abs(dot(transmitted,normal));
   let rs=(eta*cosine-transmittedCosine)/max(eta*cosine+transmittedCosine,1e-8);
   let rp=(cosine-eta*transmittedCosine)/max(cosine+eta*transmittedCosine,1e-8);
   energy*=1.-.5*(rs*rs+rp*rp);
   rd=normalize(transmitted);inside=!inside;
  }
  if(inside){energy*=exp(-vec3f(1.25,.2,.065)*.0015);}
  interfaces++;ro=p+rd*.0015;
 }
}
`
  );
}

export const causticDepositShader =
  receiverScene +
  grid +
  /* wgsl */ `
@group(0) @binding(12) var<storage,read> photons:array<Photon>;
struct VOut {@builtin(position) position:vec4f,@location(0) @interpolate(flat) energy:vec3f}
@vertex fn vertex(@builtin(vertex_index) vertex:u32)->VOut {
 var out:VOut;out.position=vec4f(2.,2.,0.,1.);out.energy=vec3f(0.);
 let triangle=vertex/3u;let cell=triangle/2u;let base=cell%(NX-1u)+NX*(cell/(NX-1u));
 let corners=select(vec3u(base,base+NX,base+1u),vec3u(base+1u,base+NX,base+NX+1u),triangle%2u==1u);
 let a=photons[corners.x];let b=photons[corners.y];let c=photons[corners.z];
 if(a.hit.w>2.5){return out;}
 if(a.hit.w<.5||b.hit.w<.5||c.hit.w<.5){return out;}
 if(a.hit.w!=b.hit.w||a.hit.w!=c.hit.w||a.entry.w!=b.entry.w||a.entry.w!=c.entry.w){return out;}
 // Do not stretch triangles across a discontinuity between water components.
 let edge=max(length(a.entry.xyz-b.entry.xyz),max(length(a.entry.xyz-c.entry.xyz),length(b.entry.xyz-c.entry.xyz)));
 if(edge>.18&&max(a.entry.w,max(b.entry.w,c.entry.w))>0.){return out;}
 let u=b.hit.xz-a.hit.xz;let v=c.hit.xz-a.hit.xz;let area=abs(u.x*v.y-u.y*v.x);
 let sourceArea=SEED_SIZE.x*SEED_SIZE.y/f32((NX-1u)*(NZ-1u));
 // Finite photon footprint avoids singular flashes at perfectly folded rays.
 let concentration=min(sourceArea/max(area,1e-8),10.);
 out.energy=(a.energy.xyz+b.energy.xyz+c.energy.xyz)*(concentration/3.);
 let hit=photons[corners[vertex%3u]].hit;
 out.position=vec4f(hit.x/RECEIVER_SIZE.x*2.,-hit.z/RECEIVER_SIZE.y*2.,0.,1.);
 return out;
}
@fragment fn fragment(in:VOut)->@location(0) vec4f {
 return vec4f(in.energy,1.);
}
`;

// Gather compact photon hits in world space at the mesh's shading vertices.
// The imported duck has mirrored/overlapping UVs, so a texture-space light map
// would incorrectly copy illumination to the opposite side of its body.
export const duckLightShader =
  grid +
  /* wgsl */ `
struct Duck {pos:vec4f,rotation:vec4f,vel:vec4f,omega:vec4f}
@group(0) @binding(0) var<uniform> control:vec4u;
@group(0) @binding(1) var<storage,read> photons:array<Photon>;
@group(0) @binding(2) var<storage,read> indices:array<u32>;
@group(0) @binding(3) var<storage,read> count:array<u32>;
@group(0) @binding(4) var<storage,read> duck:Duck;
@group(0) @binding(5) var<storage,read> triangles:array<vec4f>;
@group(0) @binding(6) var<storage,read_write> illumination:array<vec4f>;
var<workgroup> positions:array<vec4f,64>;
var<workgroup> energies:array<vec4f,64>;
var<workgroup> footprints:array<vec4f,64>;
fn rotate(q:vec4f,v:vec3f)->vec3f{return v+2.*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let vertices=control.y;if(vertices==0u){return;}
 let i=min(gid.x,vertices-1u);let triangle=i/3u;let corner=i%3u;
 let p=rotate(duck.rotation,triangles[triangle*6u+corner].xyz)+duck.pos.xyz;
 let n=normalize(rotate(duck.rotation,triangles[triangle*6u+corner+3u].xyz));
 var value=vec3f(0.);
 let sourceArea=SEED_SIZE.x*SEED_SIZE.y/f32((NX-1u)*(NZ-1u));let sunY=1./length(vec3f(-.6,1.,.35));
 for(var base=0u;base<count[1];base+=64u){
  let batchCount=min(64u,count[1]-base);
  if(lane<batchCount){let photon=photons[indices[base+lane]];positions[lane]=photon.hit;energies[lane]=photon.energy;footprints[lane]=photon.footprint;}
  workgroupBarrier();
  if(gid.x<vertices){for(var j=0u;j<batchCount;j++){
   let delta=p-positions[j].xyz;let radius=footprints[j].w;let r2=dot(delta,delta)/(radius*radius);
   // Reject nearby back-facing surfaces, rather than bleeding through the mesh.
   if(r2<1.&&dot(n,footprints[j].xyz)>.3){let q=1.-r2;value+=energies[j].rgb*(3.*sunY*sourceArea*q*q/(3.14159265*radius*radius));}
  }}
  workgroupBarrier();
 }
 if(gid.x<vertices){illumination[gid.x]=vec4f(value,1.);}
}
`;
