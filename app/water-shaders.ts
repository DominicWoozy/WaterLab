export const fullscreenVertex = `#version 300 es
in vec2 position;
out vec2 texcoord;
void main(){texcoord=position*.5+.5;gl_Position=vec4(position,0.,1.);}`;
export const particleVertex = `#version 300 es
precision highp float;
in vec3 position;
uniform vec2 resolution;
uniform vec3 eye, cameraRight, cameraUp, cameraForward;
uniform float offsetX, radius;
out float eyeDepth;
void main(){
 vec3 p=position-eye;
 float z=dot(p,cameraForward);eyeDepth=z;
 vec2 xy=vec2(dot(p,cameraRight),dot(p,cameraUp));
 float aspect=resolution.x/resolution.y;
 gl_Position=vec4(2.*1.55*xy.x/aspect-2.*offsetX*z/aspect,2.*1.55*xy.y,1.004008*z-.2004008,z);
 gl_PointSize=clamp(2.*radius*resolution.y*1.55/z,1.,160.);
}`;
export const particleFragment = `#version 300 es
precision highp float;
in float eyeDepth;
uniform float radius;
out vec4 fragColor;
void main(){
 vec2 p=gl_PointCoord*2.-1.;float r2=dot(p,p);if(r2>1.)discard;
 float h=sqrt(1.-r2),depth=eyeDepth-h*radius;
 gl_FragDepth=1.002004-.1002004/depth;
 vec3 n=vec3(p.x,-p.y,h);
 float diffuse=max(0.,dot(n,normalize(vec3(-.5,.7,1.))));
 fragColor=vec4(mix(vec3(.025,.14,.21),vec3(.32,.85,.78),diffuse),1.);
}`;
export const surfaceFragment = `#version 300 es
precision highp float;
in vec2 texcoord;
uniform highp sampler3D densityVolume, previousVolume;
uniform float fieldBlend;
uniform vec3 volumeMin, volumeMax, volumeSize, absorption;
uniform float volumeTop, isoDensity;
uniform vec2 resolution;
uniform vec3 eye,cameraRight,cameraUp,cameraForward;
uniform float offsetX,time,lightPower,reflectionOn,causticsOn,particleView;
uniform vec3 brush;
uniform float brushOn;
out vec4 fragColor;
vec3 ray(vec2 uv){vec2 q=(uv-.5)*vec2(resolution.x/resolution.y,1.);q.x+=offsetX;return normalize(cameraForward*1.55+cameraRight*q.x+cameraUp*q.y);}
float density(vec3 p){
 if(abs(p.x)>1.86||abs(p.z)>1.36||p.y<-.96||p.y>volumeTop)return 0.;
 // Map grid nodes to texel centres: CPU and GPU see the identical scalar field.
 vec3 uv=((p-volumeMin)/(volumeMax-volumeMin)*(volumeSize-1.)+.5)/volumeSize;
 float current=texture(densityVolume,uv).r;
 return fieldBlend>.999?current:mix(texture(previousVolume,uv).r,current,fieldBlend);
}
vec2 boxHit(vec3 ro,vec3 rd){
 vec3 safe=mix(vec3(.00001),rd,greaterThan(abs(rd),vec3(.00001)));
 vec3 a=(vec3(-1.87,-.97,-1.37)-ro)/safe;
 vec3 b=(vec3(1.87,volumeTop+.01,1.37)-ro)/safe;
 vec3 lo=min(a,b),hi=max(a,b);
 return vec2(max(max(lo.x,lo.y),lo.z),min(min(hi.x,hi.y),hi.z));
}
vec3 normalAt(vec3 p){
 vec3 e=vec3(.043,0.,0.);
 vec3 n=vec3(density(p-e.xyy)-density(p+e.xyy),density(p-e.yxy)-density(p+e.yxy),density(p-e.yyx)-density(p+e.yyx));
 return length(n)>.00001?normalize(n):vec3(0.,1.,0.);
}
float opticalPath(vec3 p,vec3 r){
 vec2 hit=boxHit(p,r);float end=max(0.,hit.y);
 float stepSize=end/72.;float result=0.;
 for(int i=0;i<72;i++){
  float d=density(p+r*(float(i)+.5)*stepSize);
  result+=smoothstep(isoDensity-.18,isoDensity+.18,d)*stepSize;
 }
 return result;
}
vec3 tiles(vec2 p){vec2 g=abs(fract(p*3.)-.5);float seam=smoothstep(.472,.499,max(g.x,g.y));float c=mod(floor(p.x*3.)+floor(p.y*3.),2.);return mix(mix(vec3(.65,.68,.66),vec3(.76,.78,.74),c),vec3(.12,.24,.26),seam*.75);}
vec3 sky(vec3 r){
 vec3 c=mix(vec3(.07,.18,.23),vec3(.61,.79,.83),smoothstep(-.3,.85,r.y));
 c+=vec3(.5,.62,.63)*pow(max(dot(r,normalize(vec3(-.6,.9,.3))),0.),18.);
 c+=vec3(1.,.91,.72)*pow(max(dot(r,normalize(vec3(-.6,1.,.35))),0.),260.)*2.5*lightPower;
 float stripe=smoothstep(.1,.16,r.x)*(1.-smoothstep(.25,.32,r.x))*smoothstep(.15,.35,r.y);
 c+=stripe*vec3(.85,.97,1.)*.7;
 c+=.06*sin(r.x*13.+sin(r.z*9.))*max(0.,r.y);
 return c;
}
vec3 scene(vec3 ro,vec3 rd){
 vec3 c=vec3(.045,.067,.080);
 float t=(-1.075-ro.y)/rd.y;
 if(t>0.){
  vec3 p=ro+rd*t;
  float d=length(p.xz);
  c+=vec3(.018,.026,.028)*exp(-d*.15);
  vec2 grid=abs(fract(p.xz*.5+.5)-.5);
  c+=smoothstep(.496,.5,max(grid.x,grid.y))*.014*exp(-d*.18);
  float shadow=exp(-max(abs(p.x)-1.7,0.)*3.-max(abs(p.z)-1.2,0.)*3.);
  c*=1.-shadow*.55;
 }
 float floorT=(-.97-ro.y)/rd.y;
 if(floorT>0.){
  vec3 p=ro+rd*floorT;
  if(abs(p.x)<1.92&&abs(p.z)<1.42){
   c=tiles(p.xz)*(.43+lightPower*.28);
   float rim=max(abs(p.x)/1.92,abs(p.z)/1.42);
   if(rim>.973)c=vec3(.24,.35,.37);
  }
 }
 // Transparent retaining walls give the particles a legible collision boundary.
 for(int side=0;side<4;side++){
  float axis=side<2?rd.x:rd.z;float origin=side<2?ro.x:ro.z;
  float extent=side<2?1.88:1.38;
  float at=(side==0||side==2)?-extent:extent;
  float hit=(at-origin)/axis;
  if(hit>0.){
   vec3 p=ro+rd*hit;float lateral=side<2?abs(p.z):abs(p.x);float limit=side<2?1.38:1.88;
   if(p.y>-.97&&p.y<.8&&lateral<limit){
    float line=max(exp(-abs(p.y-.8)*180.),exp(-abs(lateral-limit)*180.));
    c=mix(c,vec3(.25,.52,.54),.025+line*.45);
   }
  }
 }
 return c;
}
void main(){
 vec3 rd=ray(texcoord);vec3 color=scene(eye,rd);
 vec2 interval=boxHit(eye,rd);
 float at=max(0.,interval.x),last=at;bool found=false;
 if(particleView<.5&&interval.y>at){
  for(int i=0;i<256;i++){
   float d=density(eye+rd*at);
   if(d>isoDensity){found=true;break;}
   last=at;at+=d>.12?.02:.04;
   if(at>interval.y)break;
  }
 }
 if(found){
  for(int i=0;i<7;i++){float mid=(at+last)*.5;if(density(eye+rd*mid)>isoDensity)at=mid;else last=mid;}
  vec3 p=eye+rd*at,n=normalAt(p);
  if(dot(n,rd)>0.)n=-n;
  vec3 refracted=refract(rd,n,1./1.333);
  float thickness=opticalPath(p+refracted*.003,refracted);
  vec3 absorb=exp(-absorption*thickness);
  // Thin water is almost neutral and transparent. Colour grows with optical path length.
  vec3 transmission=scene(p+refracted*.006,refracted)*absorb;
  transmission+=vec3(.012,.11,.13)*(1.-absorb)*lightPower*.45;
  float fresnel=.0204+.9796*pow(1.-max(dot(-rd,n),0.),5.);
  color=mix(transmission,sky(reflect(rd,n)),fresnel*reflectionOn);
  vec3 sun=normalize(vec3(-.6,1.,.35));
  color+=vec3(1.,.97,.9)*pow(max(dot(reflect(rd,n),sun),0.),240.)*lightPower*reflectionOn;
  // Weak caustic accents cannot override absorption or tint thin water opaque.
  vec2 q=p.xz*10.+n.xz*1.7;
  float caustic=pow(max(0.,1.-abs(sin(q.x+sin(q.y+time*.4))+sin(q.y+sin(q.x-time*.3)))*.7),15.);
  color+=vec3(.12,.15,.14)*caustic*causticsOn*lightPower*min(thickness,.7)*absorb;
 }
 if(brushOn>.5){
  float t=(brush.y-eye.y)/rd.y;
  if(t>0.){vec3 p=eye+rd*t;float ring=exp(-pow((length(p.xz-brush.xz)-.48)*110.,2.));color+=vec3(.2,.65,.5)*ring*.55;}
 }
 color=1.-exp(-color*1.35);vec2 v=texcoord-.5;color*=1.-.12*dot(v,v);
 fragColor=vec4(pow(color,vec3(.91)),1.);
}`;
