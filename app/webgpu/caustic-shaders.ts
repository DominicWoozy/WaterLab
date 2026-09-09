import { renderScene } from './render-shaders.ts';

export const CAUSTIC_GRID = [192, 144] as const;
export const CAUSTIC_MAP = [512, 384] as const;
const grid = /* wgsl */ `
const NX=${CAUSTIC_GRID[0]}u;const NZ=${CAUSTIC_GRID[1]}u;
const SEED_SIZE=vec2f(6.,4.2);const FLOOR_SIZE=vec2f(3.84,2.84);
struct Photon {entry:vec4f,hit:vec4f,energy:vec4f}
`;

// A single air-to-water interface for the connected pool. Detached drops and
// long air gaps are rejected instead of pretending the entire path is water.
export function causticTraceShader(filterable: boolean) {
  return (
    renderScene(filterable) +
    grid +
    /* wgsl */ `
@group(0) @binding(12) var<storage,read_write> photons:array<Photon>;
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=NX||id.y>=NZ){return;}let index=id.x+NX*id.y;
 photons[index]=Photon(vec4f(0.),vec4f(0.),vec4f(0.));
 volumeTop=bitcast<f32>(bounds[0])-2.+.36;if(volumeTop<=-.9){return;}
 let sun=normalize(vec3f(-.6,1.,.35));let rd=-sun;
 let seed=(vec2f(id.xy)/vec2f(f32(NX-1u),f32(NZ-1u))-.5)*SEED_SIZE;
 // Begin above the whole duck as well as the water, including raised head shadows.
 let startY=max(volumeTop+.04,duck.pos.y+1.);
 let ro=vec3f(seed.x,-.97,seed.y)+sun*((startY+.97)/sun.y);
 let interval=boxHit(ro,rd);var t=max(0.,interval.x);let end=interval.y;
 if(end<=t){return;}var last=t;var found=false;
 for(var step=0;step<256;step++){
  if(t>end){break;}if(density(ro+rd*t)>S.config.x){found=true;break;}
  last=t;t+=.026;
 }
 if(!found){return;}
 for(var step=0;step<6;step++){
  let mid=(last+t)*.5;if(density(ro+rd*mid)>S.config.x){t=mid;}else{last=mid;}
 }
 let entry=ro+rd*t;var n=normalAt(entry);if(dot(n,rd)>0.){n=-n;}
 if(n.y<.15||duckTrace(ro,rd,t).y>=0.){return;}
 let refracted=refract(rd,n,1./1.333);if(refracted.y>=-.08){return;}
 let distance=(-.97-entry.y)/refracted.y;if(distance<=0.){return;}
 let hit=entry+refracted*distance;
 if(any(abs(hit.xz)>FLOOR_SIZE*.5)||density(hit+vec3f(0.,.07,0.))<S.config.x){return;}
 if(duckTrace(entry+refracted*.004,refracted,distance).y>=0.){return;}
 var path=0.;var gap=0.;
 for(var step=0;step<16;step++){
  let d=density(entry+refracted*(distance*(f32(step)+.5)/16.));
  let wet=smoothstep(S.config.x-.18,S.config.x+.18,d);
  path+=wet*distance/16.;gap=select(gap+distance/16.,0.,wet>.5);
  if(gap>.07){return;}
 }
 let fresnel=.0204+.9796*pow(1.-max(dot(-rd,n),0.),5.);
 let energy=(1.-fresnel)*exp(-vec3f(1.25,.2,.065)*path);
 photons[index]=Photon(vec4f(entry,1.),vec4f(hit,1.),vec4f(energy,1.));
}
`
  );
}

export const causticDepositShader =
  grid +
  /* wgsl */ `
@group(0) @binding(12) var<storage,read> photons:array<Photon>;
struct VOut {@builtin(position) position:vec4f,@location(0) @interpolate(flat) energy:vec3f}
@vertex fn vertex(@builtin(vertex_index) vertex:u32)->VOut {
 let triangle=vertex/3u;let cell=triangle/2u;let base=cell%(NX-1u)+NX*(cell/(NX-1u));
 let corners=select(vec3u(base,base+NX,base+1u),vec3u(base+1u,base+NX,base+NX+1u),triangle%2u==1u);
 let a=photons[corners.x];let b=photons[corners.y];let c=photons[corners.z];
 var out:VOut;out.position=vec4f(2.,2.,0.,1.);out.energy=vec3f(0.);
 if(a.hit.w<.5||b.hit.w<.5||c.hit.w<.5){return out;}
 // Do not stretch triangles across a discontinuity between water components.
 let edge=max(length(a.entry.xyz-b.entry.xyz),max(length(a.entry.xyz-c.entry.xyz),length(b.entry.xyz-c.entry.xyz)));
 if(edge>.18){return out;}
 let u=b.hit.xz-a.hit.xz;let v=c.hit.xz-a.hit.xz;let area=abs(u.x*v.y-u.y*v.x);
 let sourceArea=SEED_SIZE.x*SEED_SIZE.y/f32((NX-1u)*(NZ-1u));
 // Finite photon footprint avoids singular flashes at perfectly folded rays.
 let concentration=min(sourceArea/max(area,1e-8),10.);
 out.energy=(a.energy.xyz+b.energy.xyz+c.energy.xyz)*(concentration/3.);
 let hit=photons[corners[vertex%3u]].hit;
 out.position=vec4f(hit.x/FLOOR_SIZE.x*2.,-hit.z/FLOOR_SIZE.y*2.,0.,1.);
 return out;
}
@fragment fn fragment(in:VOut)->@location(0) vec4f {return vec4f(in.energy,1.);}
`;
