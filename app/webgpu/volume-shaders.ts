import { common, neighbors, cachedNeighbors } from './common.ts';
import { detailCommon } from './detail-shaders.ts';
export const shapeCommon = /* wgsl */ `
// Packed scalars: center.w=kernel radius, m0.w=amplitude, m1.w=raw rho.
// The xyz columns retain the original anisotropic metric; m2.w is Z stretch.
struct Shape {center:vec4f,m0:vec4f,m1:vec4f,m2:vec4f}
const VMIN=vec3f(-2.08,-1.12,-1.56);const VMAX=vec3f(2.08,4.08,1.56);
const VSIZE=vec3u(128,160,96);
const IDENTITY=mat3x3f(vec3f(1.,0.,0.),vec3f(0.,1.,0.),vec3f(0.,0.,1.));
fn determinant3(a:mat3x3f)->f32{return dot(a[0],cross(a[1],a[2]));}
fn inverse3(a:mat3x3f)->mat3x3f{return transpose(mat3x3f(cross(a[1],a[2]),cross(a[2],a[0]),cross(a[0],a[1])))*(1./determinant3(a));}
fn voxelIndex(p:vec3u)->u32{return p.x+128u*(p.y+160u*p.z);}
`;
const geometryNeighbor =
  'let w=q*q*q;total+=w;mean-=diff*w;cov+=mat3x3f(diff*diff.x,diff*diff.y,diff*diff.z)*w;rho+=q*q;nearby+=1.;';
export const volumeShaders: Record<string, string> = {
  geometry:
    common +
    shapeCommon +
    detailCommon +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read> input:array<Particle>;
 @group(0) @binding(2) var<storage,read_write> shapes:array<Shape>;
 @group(0) @binding(3) var<storage,read> starts:array<u32>;
 @group(0) @binding(4) var<storage,read_write> bounds:array<atomic<u32>>;
 @group(0) @binding(5) var<storage,read_write> details:array<Detail>;
 @group(0) @binding(6) var<storage,read_write> draw:array<atomic<u32>>;
 @group(0) @binding(7) var<uniform> settings:vec4u;
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=P.counts.x){return;}let p=input[i].pos.xyz;
 var total=1.;var rho=0.;var nearby=0.;var mean=vec3f(0.);var cov=IDENTITY*0.;
 ${neighbors(geometryNeighbor)}
 mean/=total;cov=cov*(1./total)-mat3x3f(mean*mean.x,mean*mean.y,mean*mean.z);
 let tr=max(cov[0][0]+cov[1][1]+cov[2][2],1e-8);
 // Small isolated primary droplets retain an analytic sub-voxel silhouette.
 // Fade this fallback out as neighbors build a continuous density surface.
 // Every primary particle always contributes to that field; there are no sheets
 // or separate secondary particles, and no hard geometry classification switch.
 let coverage=1.-smoothstep(.18,.70,rho);
 if(settings.x!=0u){
  if(i==0u){atomicStore(&draw[0],6u);}
  if(coverage>.001){
   let radius=.72*h()*pow(.1/REST,1./3.);
   let metric=IDENTITY*(1./(radius*radius));let slot=atomicAdd(&draw[1],1u);
   details[slot]=Detail(vec4f(p,1.),vec4f(metric[0],radius),vec4f(metric[1],f32(i)),vec4f(metric[2],coverage));
  }
 }
 cov+=IDENTITY*(tr*.18+1e-7);
 cov=cov*(1./pow(max(determinant3(cov),1e-24),1./3.));
 let confidence=smoothstep(5.,14.,nearby)*smoothstep(.35,1.5,rho);
 cov=IDENTITY*(1.-confidence)+cov*confidence;cov=cov*(1./pow(max(determinant3(cov),1e-8),1./3.));
 let metric=inverse3(cov);let centre=p+limited(mean*.55*confidence,.025*P.clock.z);
 // Radius and amplitude are invariant across all voxels for this particle.
 let radius=mix(.1,.19,smoothstep(.15,1.2,rho))*P.clock.z;
 shapes[i]=Shape(vec4f(centre,radius),vec4f(metric[0],1.+max(0.,1.-rho)*.8),vec4f(metric[1],rho),vec4f(metric[2],sqrt(cov[2][2])));
 atomicMax(&bounds[0],bitcast<u32>(p.y+2.));
 }`,
  density:
    common +
    shapeCommon +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read> shapes:array<Shape>;
 @group(0) @binding(2) var<storage,read> starts:array<u32>;
 @group(0) @binding(3) var<storage,read_write> field:array<f32>;
 @group(0) @binding(4) var<storage,read> bounds:array<u32>;
 @compute @workgroup_size(8,4,1) fn main(@builtin(global_invocation_id) gid:vec3u){
 if(any(gid>=VSIZE)){return;}let index=voxelIndex(gid);
 let p=VMIN+vec3f(gid)/vec3f(VSIZE-vec3u(1))*(VMAX-VMIN);
 if(P.counts.x==0u||p.y>bitcast<f32>(bounds[0])-2.+.36||p.y<-.97||abs(p.x)>1.9||abs(p.z)>1.4){field[index]=0.;return;}
 let base=cell(p);var value=0.;
 // Covariance regularization bounds max kernel stretch by cbrt(1.18/.18)<1.873.
 // Include render-centre displacement; the FINAL-position grid is rebuilt first.
 let reach=(.19*1.873+.025)*P.clock.z;let gridSize=cellSize();
 for(var z=max(0,base.z-2);z<=min(23,base.z+2);z++){
 for(var y=max(0,base.y-2);y<=min(39,base.y+2);y++){
  let lo=vec2f(-1.19,-1.53)+vec2f(f32(y),f32(z))*gridSize;
  let gap=max(max(lo-p.yz,p.yz-lo-vec2f(gridSize)),vec2f(0.));
  let remaining=reach*reach-dot(gap,gap);if(remaining<0.){continue;}
  let extent=sqrt(remaining);
  let x0=clamp(i32(floor((p.x-extent+2.04)/gridSize)),0,31);
  let x1=clamp(i32(floor((p.x+extent+2.04)/gridSize)),0,31);
  let row=32*(y+40*z);let begin=starts[u32(row+x0)];let end=starts[u32(row+x1+1)];
  for(var j=begin;j<end;j++){
   // Scale the entire kernel with particle resolution. A world-space .095
   // floor made sparse particles swell as quality increased (especially 50k).
   let centre=shapes[j].center;let radius=centre.w;
   let diff=(p-centre.xyz)/radius;if(dot(diff,diff)>3.51){continue;}
   let metric=mat3x3f(shapes[j].m0.xyz,shapes[j].m1.xyz,shapes[j].m2.xyz);
   let r2=dot(diff,metric*diff);if(r2<1.){let q=1.-r2;value+=q*q*q*shapes[j].m0.w;}
  }
 }}
 field[index]=value;
 }`,
};
for (let axis = 0; axis < 3; axis++) {
  volumeShaders[`filter${axis}`] =
    common +
    shapeCommon +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read> input:array<f32>;
 @group(0) @binding(2) var<storage,read> guide:array<f32>;
 @group(0) @binding(3) var<storage,read_write> output:array<f32>;
 fn at(p:vec3i)->u32{return voxelIndex(vec3u(clamp(p,vec3i(0),vec3i(VSIZE)-vec3i(1))));}
 @compute @workgroup_size(8,4,4) fn main(@builtin(global_invocation_id) gid:vec3u){
 if(any(gid>=VSIZE)){return;}let index=voxelIndex(gid);let p=vec3i(gid);let centre=guide[index];
 if(centre<.02){output[index]=centre;return;}
 var direction=vec3i(0);direction[${axis}]=1;var sum=0.;var peak=0.;
 for(var k=-2;k<=2;k++){let j=at(p+direction*k);var w=1.;if(abs(k)==1){w=4.;}if(k==0){w=6.;}sum+=input[j]*w;peak=max(peak,guide[j]);}
 for(var a=0;a<3;a++){var delta=vec3i(0);delta[a]=2;peak=max(peak,max(guide[at(p+delta)],guide[at(p-delta)]));}
 let gradient=vec3f(guide[at(p+vec3i(1,0,0))]-guide[at(p-vec3i(1,0,0))],guide[at(p+vec3i(0,1,0))]-guide[at(p-vec3i(0,1,0))],guide[at(p+vec3i(0,0,1))]-guide[at(p-vec3i(0,0,1))]);
 let nw=gradient[${axis}]*gradient[${axis}]/max(dot(gradient,gradient),1e-8);
 let blend=smoothstep(1.85,4.,peak)*.85*(1.-.95*nw);output[index]=mix(centre,sum/16.,blend);
 }`;
}

// Velocity projection does not move or reorder particles. Its final exact
// neighbor cache is also valid for the identical geometry kernel radius.
// Overflow still uses complete grid traversal, never a truncated list.
volumeShaders.geometryCached = volumeShaders.geometry.replace(
  neighbors(geometryNeighbor),
  cachedNeighbors(geometryNeighbor),
);
