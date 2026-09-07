import { PARTICLE_WIDTH } from './gpu-particle-config.ts';
import {
  GPU_VOLUME_SIZE,
  GPU_ATLAS_SIZE,
  GPU_SLICES_PER_PARTICLE,
  BULK_KERNEL_RADIUS,
  SPRAY_KERNEL_RADIUS,
} from './gpu-volume-config.ts';
/** WebGL 2 fragment-compute passes. Particle state never leaves GPU memory. */
export const computeVertex = `#version 300 es
precision highp float;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0.,1.);}`;
const common = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D positions, velocities, sortedKeys, cellRanges, lambdas, oldPositions;
uniform int count;
uniform float particleScale;
const float REST=3.6;
#define H (.17*particleScale)
ivec2 uv(int i){return ivec2(i%${PARTICLE_WIDTH},i/${PARTICLE_WIDTH});}
int id(){return int(gl_FragCoord.x)+int(gl_FragCoord.y)*${PARTICLE_WIDTH};}
vec4 readAt(sampler2D t,int i){return texelFetch(t,uv(i),0);}
ivec3 cell(vec3 p){return clamp(ivec3(floor((p-vec3(-2.04,-1.19,-1.53))/(.225*particleScale))),ivec3(0),ivec3(31,39,23));}
int key(ivec3 c){return c.x+32*(c.y+40*c.z);}
vec3 bound(vec3 p){return clamp(p,vec3(-1.78,-.917,-1.28),vec3(1.78,3.8,1.28));}
vec3 limited(vec3 v,float m){return v*min(1.,m/max(length(v),.000001));}
`;
export const initializeFragment =
  common +
  `
out vec4 result;
uniform int initialCount;
void main(){int i=id();int nx=initialCount>15000?75:60,nz=50,ny=initialCount>15000?8:5;
 result=i<count?vec4(-1.7+float(i%nx)*3.4/float(nx-1),-.917+float(i/(nx*nz))*.36/float(ny-1),-1.2+float((i/nx)%nz)*2.4/float(nz-1),1.):vec4(0.); }`;
export const predictFragment =
  common +
  `
uniform float dt, time, gravity, agitation, shake;
uniform vec4 brush; // xyz position, w strength; zero strength means inactive
uniform vec2 brushVelocity, pourAt;
uniform vec3 splash; // x,z,strength
uniform int previousCount;
out vec4 result;
float random(float x){return fract(sin(x*127.1+time*311.7)*43758.5453);}
void main(){
 int i=id();if(i>=count){result=vec4(0.);return;}
 vec3 p=readAt(positions,i).xyz,v=readAt(velocities,i).xyz;
 if(i>=previousCount){float a=float(i);p=vec3(pourAt.x+(random(a)-.5)*.29,1.8+random(a+33.)*.25,pourAt.y+(random(a+61.)-.5)*.29);v=vec3(0.,-1.4,0.);}
 v.y-=gravity*dt;
 v.x+=(shake+sin(time*2.1+p.z*3.)*agitation*2.)*dt;
 v.z+=cos(time*1.7+p.x*2.)*agitation*dt;
 vec3 d=p-brush.xyz;float w=exp(-dot(d,d)/.32)*brush.w;
 v+=vec3(brushVelocity.x-d.z*3.,.65,brushVelocity.y+d.x*3.)*w*dt*12.;
 vec2 sd=p.xz-splash.xy;float sw=exp(-dot(sd,sd)/.18)*splash.z;
 v+=vec3(sd.x*2.,1.6,sd.y*2.)*sw;
 result=vec4(bound(p+limited(v,12.)*dt),1.);
}`;
export const keyFragment =
  common +
  `
out vec4 result;
void main(){int i=id();result=vec4(i<count?float(key(cell(readAt(positions,i).xyz))):1e9,float(i),0.,0.);}`;
export const sortFragment =
  common +
  `
uniform int stage, stride;
out vec4 result;
void main(){int i=id(),j=i^stride;vec4 a=readAt(sortedKeys,i),b=readAt(sortedKeys,j);
 bool less=a.x<b.x||(a.x==b.x&&a.y<b.y);
 bool ascending=(i&stage)==0;bool lower=(i&stride)==0;
 result=(less==(ascending==lower))?a:b;}`;
export const rangesFragment =
  common +
  `
uniform int sortCount;
out vec4 result;
int lowerBound(float k){int lo=0,hi=sortCount;for(int n=0;n<16;n++){if(lo>=hi)break;int m=(lo+hi)/2;if(readAt(sortedKeys,m).x<k)lo=m+1;else hi=m;}return lo;}
void main(){int k=id();result=vec4(float(lowerBound(float(k))),float(lowerBound(float(k+1))),0.,0.);}`;
/** Spatially reorder all coupled state together (Green 2010 / Hoetzlein 2014).
 * New-particle tags follow the permutation, rather than assuming new slots stay at the end. */
export const reorderFragment =
  common +
  `
uniform int previousCount;
layout(location=0) out vec4 nextPosition;
layout(location=1) out vec4 previousPosition;
layout(location=2) out vec4 nextVelocity;
void main(){
 int i=id();if(i>=count){nextPosition=vec4(0.);previousPosition=vec4(0.);nextVelocity=vec4(0.);return;}
 int j=int(readAt(sortedKeys,i).y);
 nextPosition=readAt(positions,j);
 previousPosition=vec4(readAt(oldPositions,j).xyz,j<previousCount?1.:0.);
 nextVelocity=readAt(velocities,j);
}`;
// Ranges contain every particle in a cell: no fixed neighbour bucket or overflow truncation.
const neighbors = (body: string) => `
 ivec3 base=cell(p);
 for(int z=-1;z<=1;z++)for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
  ivec3 c=base+ivec3(x,y,z);if(any(lessThan(c,ivec3(0)))||any(greaterThan(c,ivec3(31,39,23))))continue;
  // A neighbour may have moved by at most three correction clamps since grid build.
  // Use that conservative reach so culling cannot drop a valid corrected neighbour.
  vec3 lo=vec3(-2.04,-1.19,-1.53)+vec3(c)*(.225*particleScale);
  vec3 hi=lo+vec3(.225*particleScale);
  vec3 gap=max(max(lo-p,p-hi),vec3(0.));
  if(dot(gap,gap)>pow(H+.051*particleScale,2.))continue;
  vec2 range=readAt(cellRanges,key(c)).xy;
  for(int at=int(range.x);at<int(range.y);at++){
   int j=at;if(j==i)continue;
   vec3 d=p-readAt(positions,j).xyz;float r2=dot(d,d);if(r2>=H*H||r2<1e-12)continue;float r=sqrt(r2);
   float q=1.-r/H;vec3 gradient=(2.*q/(H*REST*r))*d;
   ${body}
  }
 }
`;
export const lambdaFragment =
  common +
  `
out vec4 result;
void main(){int i=id();if(i>=count){result=vec4(0.);return;}vec3 p=readAt(positions,i).xyz;
 float rho=0.,sum=0.;vec3 grad=vec3(0.);
 ${neighbors('rho+=q*q; grad+=gradient; sum+=dot(gradient,gradient);')}
 result=vec4(-max(rho/REST-1.,0.)/(sum+dot(grad,grad)+2.),rho,0.,0.);
}`;
export const correctFragment =
  common +
  `
out vec4 result;
void main(){int i=id();if(i>=count){result=vec4(0.);return;}vec3 p=readAt(positions,i).xyz;
 float lambda=readAt(lambdas,i).x;vec3 delta=vec3(0.);
 ${neighbors('delta-=(lambda+readAt(lambdas,j).x)*gradient;')}
 result=vec4(bound(p+limited(delta,.017*particleScale)),1.);
}`;
export const velocityFragment =
  common +
  `
uniform float dt;
uniform int previousCount;
out vec4 result;
void main(){int i=id();if(i>=count){result=vec4(0.);return;}
 vec3 v=(readAt(positions,i).xyz-readAt(oldPositions,i).xyz)/dt;
 if(readAt(oldPositions,i).w<.5)v=vec3(0.,-1.4,0.);
 result=vec4(limited(v,12.)*.998,0.);}`;
export const viscosityFragment =
  common +
  `
uniform float viscosity;
out vec4 result;
void main(){int i=id();if(i>=count){result=vec4(0.);return;}vec3 p=readAt(positions,i).xyz,v=readAt(velocities,i).xyz,delta=vec3(0.);
 ${neighbors('delta+=(readAt(velocities,j).xyz-v)*q*q;')}
 result=vec4(limited(v+delta*(.002+viscosity*.065),12.),0.);
}`;
/** DFSPH velocity projection (Bender & Koschier, equations 9–11).
 * Equal-mass, normalized-density formulation; positive compression only.
 * The position solver remains PBF: this is a hybrid, not the full DFSPH integrator. */
export const divergenceFactorFragment =
  common +
  `
out vec4 result;
void main(){int i=id();if(i>=count){result=vec4(0.);return;}vec3 p=readAt(positions,i).xyz;
 float rho=0.,sum=0.,nearby=0.;vec3 grad=vec3(0.);
 ${neighbors('rho+=q*q;grad+=gradient;sum+=dot(gradient,gradient);nearby+=1.;')}
 // Sparse ballistic spray has no reliable divergence estimate.
 float factor=nearby>=12.&&rho>REST*.4?1./max(sum+dot(grad,grad),1e-6):0.;
 result=vec4(factor,rho/REST,nearby,0.);
}`;
export const divergenceResidualFragment =
  common +
  `
uniform sampler2D factors;
out vec4 result;
void main(){int i=id();if(i>=count){result=vec4(0.);return;}vec3 p=readAt(positions,i).xyz,v=readAt(velocities,i).xyz;
 float rate=0.;
 ${neighbors('rate+=dot(readAt(velocities,j).xyz-v,gradient);')}
 vec4 f=readAt(factors,i);
 result=vec4(max(rate,0.)*f.x,rate,f.y,0.);
}`;
export const divergenceProjectFragment =
  common +
  `
out vec4 result;
void main(){int i=id();if(i>=count){result=vec4(0.);return;}vec3 p=readAt(positions,i).xyz,v=readAt(velocities,i).xyz;
 float pressure=readAt(lambdas,i).x;vec3 delta=vec3(0.);
 ${neighbors('delta+=(pressure+readAt(lambdas,j).x)*gradient;')}
 // Relaxed Jacobi, with a non-penetrating slip boundary after projection.
 v=limited(v+.5*delta,12.);
 if(p.x<=-1.77999)v.x=max(v.x,0.);if(p.x>=1.77999)v.x=min(v.x,0.);
 if(p.y<=-.91699)v.y=max(v.y,0.);if(p.y>=3.79999)v.y=min(v.y,0.);
 if(p.z<=-1.27999)v.z=max(v.z,0.);if(p.z>=1.27999)v.z=min(v.z,0.);
 result=vec4(v,0.);
}`;
/** Yu & Turk inspired covariance kernels and render-only centre smoothing.
 * Regularization replaces their explicit eigenvalue clamp; not a verbatim paper implementation. */
export const geometryFragment =
  common +
  `
layout(location=0) out vec4 center;
layout(location=1) out vec4 metric0;
layout(location=2) out vec4 metric1;
layout(location=3) out vec4 metric2;
void main(){
 int i=id();vec3 p=readAt(positions,i).xyz;
 float total=1.,rho=0.,nearby=0.;vec3 mean=vec3(0.);mat3 cov=mat3(0.);
 if(i<count){
 ${neighbors('float w=q*q*q;total+=w;mean-=d*w;cov+=outerProduct(d,d)*w;rho+=q*q;nearby+=1.;')}
 }
 mean/=total;cov=cov/total-outerProduct(mean,mean);
 float trace=max(cov[0][0]+cov[1][1]+cov[2][2],1e-8);
 // Positive definite regularization bounds the axis ratio, including sheets and streams.
 cov+=mat3(trace*.18+1e-7);
 cov/=pow(max(determinant(cov),1e-24),1./3.);
 float confidence=smoothstep(5.,14.,nearby)*smoothstep(.35,1.5,rho);
 cov=mat3(1.)*(1.-confidence)+cov*confidence;
 cov/=pow(max(determinant(cov),1e-8),1./3.);
 mat3 metric=inverse(cov);
 center=vec4(p+limited(mean*.55*confidence,.025*particleScale),rho);
 metric0=vec4(metric[0],sqrt(cov[0][0]));metric1=vec4(metric[1],sqrt(cov[1][1]));metric2=vec4(metric[2],sqrt(cov[2][2]));
}
`;
// 96 Z slices in an 8x12 atlas. Each particle emits only the slices its kernel intersects.
export const volumeVertex = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D positions, lambdas, metric0, metric1, metric2;
uniform float particleScale;
out vec3 local;
flat out mat3 metric;
out float weight;
const vec3 lo=vec3(-2.08,-1.12,-1.56),hi=vec3(2.08,4.08,1.56),size=vec3(${GPU_VOLUME_SIZE.map((n) => n.toFixed(1)).join(',')});
void main(){
 int i=gl_InstanceID/${GPU_SLICES_PER_PARTICLE},k=gl_InstanceID%${GPU_SLICES_PER_PARTICLE};ivec2 uv=ivec2(i%${PARTICLE_WIDTH},i/${PARTICLE_WIDTH});
 vec4 center=texelFetch(positions,uv,0);vec3 p=center.xyz;float rho=center.w;
 vec4 m0=texelFetch(metric0,uv,0),m1=texelFetch(metric1,uv,0),m2=texelFetch(metric2,uv,0);
 metric=mat3(m0.xyz,m1.xyz,m2.xyz);
 float bulk=smoothstep(.15,1.2,rho);
 // Detached spray gets a smaller support; connected water keeps its broad smooth surface.
 float radius=max(.095,mix(${SPRAY_KERNEL_RADIUS.toFixed(3)},${BULK_KERNEL_RADIUS.toFixed(3)},bulk)*particleScale);
 // Bounds were computed once per particle, rather than in all 120 splat vertices.
 vec3 extent=radius*vec3(m0.w,m1.w,m2.w);
 int slice=int(ceil((p.z-extent.z-lo.z)/(hi.z-lo.z)*(size.z-1.)))+k;
 vec2 corners[6]=vec2[6](vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(-1.,1.),vec2(1.,-1.),vec2(1.,1.));
 vec2 xy=p.xy+corners[gl_VertexID]*extent.xy;
 vec2 node=(xy-lo.xy)/(hi.xy-lo.xy)*(size.xy-1.);
 vec2 tile=vec2(slice%8,slice/8);
 vec2 pixel=tile*size.xy+node+.5;
 gl_Position=vec4(pixel/vec2(${GPU_ATLAS_SIZE.map((n) => n.toFixed(1)).join(',')})*2.-1.,0.,1.);
 float z=lo.z+float(slice)/(size.z-1.)*(hi.z-lo.z);
 local=vec3(xy-p.xy,z-p.z)/radius;
 weight=1.+max(0.,1.-rho)*.8;
 if(slice<0||slice>=int(size.z)||abs(z-p.z)>extent.z)gl_Position=vec4(2.,2.,2.,1.);
}`;
export const volumeFragment = `#version 300 es
precision highp float;
in vec3 local;
in float weight;
flat in mat3 metric;
out vec4 result;
void main(){float r2=dot(local,metric*local);if(r2>=1.)discard;float q=1.-r2;result=vec4(q*q*q*weight,0.,0.,1.);}`;
/** Separable, world-space reconstruction filter. A raw-density guide disables it
 * for isolated spray (whose kernel peak is <= 1.8), preserving small droplets.
 * Atlas neighbours are addressed in 3D: never sample across adjacent atlas tiles. */
export const surfaceFilterFragment = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D source, guide;
uniform vec3 axis;
out vec4 result;
ivec2 atlasUV(ivec3 p){
 p=clamp(p,ivec3(0),ivec3(127,159,95));
 return ivec2((p.z%8)*128+p.x,(p.z/8)*160+p.y);
}
void main(){
 ivec2 pixel=ivec2(gl_FragCoord.xy);
 ivec3 p=ivec3(pixel.x%128,pixel.y%160,(pixel.x/128)+(pixel.y/160)*8);
 float center=texelFetch(source,pixel,0).r;
 // Most of the atlas is air. Preserve its empty support and skip the stencil.
 if(center<.02){result=vec4(center,0.,0.,1.);return;}
 float sum=0.,peak=0.;
 for(int k=-2;k<=2;k++){
  ivec2 at=atlasUV(p+ivec3(axis)*k);
  float w=k==0?6.:(abs(k)==1?4.:1.);
  sum+=texelFetch(source,at,0).r*w;
  peak=max(peak,texelFetch(guide,at,0).r);
 }
 // Connectivity must be independent of the current filter axis; otherwise a
 // flat horizontal surface would be smoothed vertically but not tangentially.
 for(int a=0;a<3;a++){
  ivec3 offset=ivec3(0);offset[a]=2;
  peak=max(peak,max(texelFetch(guide,atlasUV(p+offset),0).r,texelFetch(guide,atlasUV(p-offset),0).r));
 }
 vec3 gradient=vec3(
  texelFetch(guide,atlasUV(p+ivec3(1,0,0)),0).r-texelFetch(guide,atlasUV(p-ivec3(1,0,0)),0).r,
  texelFetch(guide,atlasUV(p+ivec3(0,1,0)),0).r-texelFetch(guide,atlasUV(p-ivec3(0,1,0)),0).r,
  texelFetch(guide,atlasUV(p+ivec3(0,0,1)),0).r-texelFetch(guide,atlasUV(p-ivec3(0,0,1)),0).r);
 // Smooth mainly along the surface, limiting expansion across its normal.
 float normalWeight=pow(dot(gradient,axis),2.)/max(dot(gradient,gradient),1e-8);
 float blend=smoothstep(1.85,4.,peak)*.85*(1.-.95*normalWeight);
 result=vec4(mix(center,sum/16.,blend),0.,0.,1.);
}`;
/** Parallel max reduction provides tight ray bounds without a CPU readback. */
export const boundsFragment = `#version 300 es
precision highp float;
uniform highp sampler2D source;
uniform int count;
uniform float particleScale;
uniform float firstLevel;
out vec4 result;
void main(){
 ivec2 p=ivec2(gl_FragCoord.xy)*2;float top=-1.2;
 for(int y=0;y<2;y++)for(int x=0;x<2;x++){
  ivec2 at=min(p+ivec2(x,y),textureSize(source,0)-1);
  vec4 a=texelFetch(source,at,0);
  if(a.w>.5&&(firstLevel<.5||at.x+at.y*${PARTICLE_WIDTH}<count))top=max(top,a.y);
 }
 result=vec4(0.,top,0.,1.);
}`;
