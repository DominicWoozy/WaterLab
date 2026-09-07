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
uniform float radius, thicknessPass;
out vec4 fragColor;
void main(){
 vec2 p=gl_PointCoord*2.-1.;float r2=dot(p,p);if(r2>1.)discard;
 float h=sqrt(1.-r2);float depth=eyeDepth-h*radius;
 gl_FragDepth=1.002004-.1002004/depth;
 if(thicknessPass>.5)fragColor=vec4(vec3(h*.035),1.);
 else fragColor=vec4(depth,0.,0.,1.);
}`;
export const blurFragment = `#version 300 es
precision highp float;
in vec2 texcoord;
uniform sampler2D source;
uniform vec2 direction,resolution;
out vec4 fragColor;
void main(){
 float center=texture(source,texcoord).r;
 if(center==0.){fragColor=vec4(0.);return;}
 float sum=0.,weight=0.;
 float scale=clamp(resolution.y/center/140.,.6,1.5);
 for(int i=-7;i<=7;i++){
  float d=texture(source,texcoord+direction*float(i)*scale/resolution).r;
  if(d==0.)continue;
  float w=exp(-float(i*i)/22.-pow((d-center)*4.2,2.));
  sum+=d*w;weight+=w;
 }
 fragColor=vec4(sum/max(weight,.0001),0.,0.,1.);
}`;
export const surfaceFragment = `#version 300 es
precision highp float;
in vec2 texcoord;
uniform sampler2D depthMap, thicknessMap;
uniform vec2 resolution;
uniform vec3 eye,cameraRight,cameraUp,cameraForward;
uniform float offsetX,time,lightPower,reflectionOn,causticsOn,particleView;
uniform vec3 brush;
uniform float brushOn;
out vec4 fragColor;
vec3 ray(vec2 uv){vec2 q=(uv-.5)*vec2(resolution.x/resolution.y,1.);q.x+=offsetX;return normalize(cameraForward*1.55+cameraRight*q.x+cameraUp*q.y);}
vec3 world(vec2 uv,float depth){vec3 r=ray(uv);return eye+r*depth/max(dot(r,cameraForward),.001);}
vec3 tiles(vec2 p){vec2 g=abs(fract(p*3.)-.5);float seam=smoothstep(.472,.499,max(g.x,g.y));float c=mod(floor(p.x*3.)+floor(p.y*3.),2.);return mix(mix(vec3(.36,.49,.49),vec3(.48,.59,.57),c),vec3(.12,.24,.26),seam*.75);}
vec3 sky(vec3 r){
 vec3 c=mix(vec3(.07,.18,.23),vec3(.61,.79,.83),smoothstep(-.3,.85,r.y));
 c+=vec3(.5,.62,.63)*pow(max(dot(r,normalize(vec3(-.6,.9,.3))),0.),18.);
 c+=vec3(1.,.91,.72)*pow(max(dot(r,normalize(vec3(-.6,1.,.35))),0.),260.)*2.5*lightPower;
 float stripe=smoothstep(.1,.16,r.x)*smoothstep(.32,.25,r.x)*smoothstep(.15,.35,r.y);
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
   c=tiles(p.xz)*(.27+lightPower*.23);
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
 vec3 rd=ray(texcoord);float d=texture(depthMap,texcoord).r;
 vec3 color=scene(eye,rd);
 if(d>0.){
  vec3 p=world(texcoord,d);
  vec2 e=1./resolution;
  float dl=texture(depthMap,texcoord-vec2(e.x,0.)).r;
  float dr=texture(depthMap,texcoord+vec2(e.x,0.)).r;
  float db=texture(depthMap,texcoord-vec2(0.,e.y)).r;
  float dt=texture(depthMap,texcoord+vec2(0.,e.y)).r;
  if(dl==0.)dl=d+2.;if(dr==0.)dr=d+2.;if(db==0.)db=d+2.;if(dt==0.)dt=d+2.;
  vec3 dx=abs(dr-d)<abs(dl-d)?world(texcoord+vec2(e.x,0.),dr)-p:p-world(texcoord-vec2(e.x,0.),dl);
  vec3 dy=abs(dt-d)<abs(db-d)?world(texcoord+vec2(0.,e.y),dt)-p:p-world(texcoord-vec2(0.,e.y),db);
  vec3 normal=cross(dx,dy);vec3 n=length(normal)>.00000001?normalize(normal):-rd;
  if(dot(n,rd)>0.)n=-n;
  float thickness=0.;
  for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++)thickness+=texture(thicknessMap,texcoord+vec2(x,y)*e*2.).r;
  thickness=max(.055,thickness/9.*3.8);
  vec3 refracted=refract(rd,n,1./1.333);
  vec3 transmission=scene(p+refracted*.025,refracted)*1.65;
  vec3 absorb=exp(-vec3(1.9,.42,.23)*thickness);
  transmission=transmission*absorb+vec3(.02,.34,.39)*(1.-absorb)*(.5+lightPower*.28);
  // Moving refractive highlight approximation, driven by reconstructed particle normals.
  vec2 q=p.xz*11.+n.xz*2.;
  float caustic=pow(max(0.,1.-abs(sin(q.x+sin(q.y+time*.4))+sin(q.y+sin(q.x-time*.3)))*.7),12.);
  transmission+=vec3(.11,.24,.22)*caustic*causticsOn*lightPower*(1.-absorb.r);
  float fresnel=.025+.975*pow(1.-max(dot(-rd,n),0.),5.);
  vec3 reflected=sky(reflect(rd,n));
  color=mix(transmission,reflected,fresnel*reflectionOn);
  vec3 sun=normalize(vec3(-.6,1.,.35));
  color+=vec3(1.,.96,.83)*pow(max(dot(reflect(rd,n),sun),0.),130.)*lightPower*reflectionOn*1.7;
  if(particleView>.5)color=mix(vec3(.025,.21,.28),vec3(.26,.87,.78),max(0.,dot(n,sun))*.8+.2)+pow(max(dot(reflect(rd,n),sun),0.),45.)*.6;
 }
 if(brushOn>.5){
  float t=(brush.y-eye.y)/rd.y;
  if(t>0.){vec3 p=eye+rd*t;float r=length(p.xz-brush.xz);float ring=exp(-pow((r-.48)*110.,2.));color+=vec3(.2,.65,.5)*ring*.55;}
 }
 color=1.-exp(-color*1.35);
 vec2 v=texcoord-.5; color*=1.-.12*dot(v,v);
 fragColor=vec4(pow(color,vec3(.91)),1.);
}`;
