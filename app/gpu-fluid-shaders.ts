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
const float H=.17, REST=3.6;
ivec2 uv(int i){return ivec2(i%128,i/128);}
int id(){return int(gl_FragCoord.x)+int(gl_FragCoord.y)*128;}
vec4 readAt(sampler2D t,int i){return texelFetch(t,uv(i),0);}
ivec3 cell(vec3 p){return clamp(ivec3(floor((p-vec3(-2.04,-1.19,-1.53))/.21)),ivec3(0),ivec3(25,31,20));}
int key(ivec3 c){return c.x+26*(c.y+32*c.z);}
vec3 bound(vec3 p){return clamp(p,vec3(-1.78,-.917,-1.28),vec3(1.78,3.8,1.28));}
vec3 limited(vec3 v,float m){return v*min(1.,m/max(length(v),.000001));}
`;
export const initializeFragment =
  common +
  `
out vec4 result;
void main(){int i=id();result=i<count?vec4(-1.7+float(i%50)*(3.4/49.),-.917+float(i/2000)*.09,-1.2+float((i/50)%40)*(2.4/39.),1.):vec4(0.);}`;
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
out vec4 result;
int lowerBound(float k){int lo=0,hi=16384;for(int n=0;n<15;n++){if(lo>=hi)break;int m=(lo+hi)/2;if(readAt(sortedKeys,m).x<k)lo=m+1;else hi=m;}return lo;}
void main(){int k=id();result=vec4(float(lowerBound(float(k))),float(lowerBound(float(k+1))),0.,0.);}`;
// Ranges contain every particle in a cell: no fixed neighbour bucket or overflow truncation.
const neighbors = (body: string) => `
 ivec3 base=cell(p);
 for(int z=-1;z<=1;z++)for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
  ivec3 c=base+ivec3(x,y,z);if(any(lessThan(c,ivec3(0)))||any(greaterThan(c,ivec3(25,31,20))))continue;
  vec2 range=readAt(cellRanges,key(c)).xy;
  for(int at=int(range.x);at<int(range.y);at++){
   int j=int(readAt(sortedKeys,at).y);if(j==i)continue;
   vec3 d=p-readAt(positions,j).xyz;float r=length(d);if(r>=H||r<.000001)continue;
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
 result=vec4(bound(p+limited(delta,.017)),1.);
}`;
export const velocityFragment =
  common +
  `
uniform float dt;
uniform int previousCount;
out vec4 result;
void main(){int i=id();if(i>=count){result=vec4(0.);return;}
 vec3 v=(readAt(positions,i).xyz-readAt(oldPositions,i).xyz)/dt;
 if(i>=previousCount)v=vec3(0.,-1.4,0.);
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
// 72 Z slices in an 8x9 atlas. Each particle emits only the slices its kernel intersects.
export const volumeVertex = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D positions, lambdas;
out vec3 local;
out float weight;
const vec3 lo=vec3(-2.08,-1.12,-1.56),hi=vec3(2.08,4.08,1.56),size=vec3(96.,128.,72.);
void main(){
 int i=gl_InstanceID/10,k=gl_InstanceID%10;ivec2 uv=ivec2(i%128,i/128);
 vec3 p=texelFetch(positions,uv,0).xyz;
 int slice=int(ceil((p.z-.19-lo.z)/(hi.z-lo.z)*71.))+k;
 vec2 corners[6]=vec2[6](vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(-1.,1.),vec2(1.,-1.),vec2(1.,1.));
 vec2 xy=p.xy+corners[gl_VertexID]*.19;
 vec2 node=(xy-lo.xy)/(hi.xy-lo.xy)*(size.xy-1.);
 vec2 tile=vec2(slice%8,slice/8);
 vec2 pixel=tile*size.xy+node+.5;
 gl_Position=vec4(pixel/vec2(768.,1152.)*2.-1.,0.,1.);
 float z=lo.z+float(slice)/71.*(hi.z-lo.z);
 local=vec3(xy-p.xy,z-p.z)/.19;
 weight=1.+max(0.,1.-texelFetch(lambdas,uv,0).y)*.8;
 if(slice<0||slice>=72||abs(z-p.z)>.19)gl_Position=vec4(2.,2.,2.,1.);
}`;
export const volumeFragment = `#version 300 es
precision highp float;
in vec3 local;
in float weight;
out vec4 result;
void main(){float r2=dot(local,local);if(r2>=1.)discard;float q=1.-r2;result=vec4(q*q*q*weight,0.,0.,1.);}`;
/** Parallel max reduction provides tight ray bounds without a CPU readback. */
export const boundsFragment = `#version 300 es
precision highp float;
uniform highp sampler2D source;
uniform int count;
uniform float firstLevel;
out vec4 result;
void main(){
 ivec2 p=ivec2(gl_FragCoord.xy)*2;float top=-1.2;
 for(int y=0;y<2;y++)for(int x=0;x<2;x++){
  ivec2 at=p+ivec2(x,y);
  vec4 a=texelFetch(source,at,0);
  if(a.w>.5&&(firstLevel<.5||at.x+at.y*128<count))top=max(top,a.y);
 }
 result=vec4(0.,top,0.,1.);
}`;
