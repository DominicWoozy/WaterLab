import { common, cachedNeighbors as neighbors } from './common.ts';
import { tensionCommon } from './surface-tension-shaders.ts';

// Frozen-coefficient backward Euler, inspired by Jeske et al.'s separation of
// local coefficients from the implicit velocity solve. This is not their full
// density-derivative/implicit-viscosity formulation. The repulsive core stays
// explicit; the positive cohesion graph gives a diagonally dominant system.
const bindings = /* wgsl */ `
@group(0) @binding(1) var<storage,read> input:array<Particle>;
@group(0) @binding(2) var<storage,read_write> output:array<Particle>;
@group(0) @binding(3) var<storage,read> starts:array<u32>;
@group(0) @binding(4) var<storage,read> surface:array<vec4f>;
@group(0) @binding(5) var<storage,read_write> rhs:array<vec4f>;
@group(0) @binding(6) var<storage,read> guess:array<vec4f>;
@group(0) @binding(7) var<storage,read_write> next:array<vec4f>;
@group(0) @binding(8) var<storage,read> factors:array<vec4f>;
`;
const kernel = (body: string) =>
  common +
  tensionCommon +
  bindings +
  /* wgsl */ `
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=P.counts.x){return;}${body}
}`;
export const capillaryShaders = {
  capillaryPrepare: kernel(/* wgsl */ `
 let p=input[i].pos.xyz;let v=input[i].vel.xyz;var force=vec3f(0.);var diagonal=1.;
 {
  ${neighbors('let f=surfacePair(surface[i],surface[j],diff,r,factors[i].z,factors[j].z);force-=diff*(f/r);diagonal+=P.clock.x*P.clock.x*max(f/r,0.);')}
 }
 rhs[i]=vec4f(v+P.clock.x*force,1./diagonal);next[i]=vec4f(v,0.);
 `),
  capillaryIterate: kernel(/* wgsl */ `
 let p=input[i].pos.xyz;var sum=vec3f(0.);
 {
  ${neighbors('let k=P.clock.x*P.clock.x*max(surfacePair(surface[i],surface[j],diff,r,factors[i].z,factors[j].z)/r,0.);sum+=k*guess[j].xyz;')}
 }
 let v=(rhs[i].xyz+sum)*rhs[i].w;
 next[i]=vec4f(v,length(v-guess[i].xyz));
 `),
  capillaryApply: kernel(/* wgsl */ `
 var a=input[i];a.vel=vec4f(limited(guess[i].xyz,12.),a.vel.w);output[i]=a;
 `),
};
