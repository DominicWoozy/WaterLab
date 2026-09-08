/** A sealed, bottom-weighted rubber duck: metres, seconds, water density = 1.
 * State stays on the GPU: centre of mass, unit quaternion, linear/angular velocity.
 * Collision hull and imported Sony mesh share a local coordinate system. The ellipsoids are a collision approximation.
 */
export const duckCommon = `
uniform highp sampler2D duckState;
vec3 duckPosition(){return texelFetch(duckState,ivec2(0,0),0).xyz;}
vec4 duckRotation(){return texelFetch(duckState,ivec2(1,0),0);}
vec3 qrotate(vec4 q,vec3 v){return v+2.*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
vec3 duckLocal(vec3 p){vec4 q=duckRotation();return qrotate(vec4(-q.xyz,q.w),p-duckPosition());}
vec3 duckVelocity(vec3 p){return texelFetch(duckState,ivec2(2,0),0).xyz+cross(texelFetch(duckState,ivec2(3,0),0).xyz,p-duckPosition());}
const float DUCK_MASS=.026;
const vec3 DUCK_INERTIA=vec3(.00105,.00075,.00145);
float ellipsoid(vec3 p,vec3 r){return (length(p/r)-1.)*min(r.x,min(r.y,r.z));}
// Signed distance estimate + outward world normal for the sealed collision hull.
vec4 duckHull(vec3 p){
 vec3 x=duckLocal(p),r=vec3(.34,.195,.25),d=x-vec3(.025,.055,0.);
 float best=ellipsoid(d,r);
 vec3 nr=vec3(.135,.20,.135),nd=x-vec3(.17,.24,0.);float n=ellipsoid(nd,nr);
 if(n<best){best=n;d=nd;r=nr;}
 nr=vec3(.21,.16,.18);nd=x-vec3(.19,.39,0.);n=ellipsoid(nd,nr);
 if(n<best){best=n;d=nd;r=nr;}
 vec3 normal=qrotate(duckRotation(),normalize(d/(r*r)+vec3(1e-8)));
 return vec4(normal,best);
}
vec3 duckInverseInertia(vec3 torque){
 vec4 q=duckRotation();return qrotate(q,qrotate(vec4(-q.xyz,q.w),torque)/DUCK_INERTIA);
}
`;

export const duckInitializeFragment = `#version 300 es
precision highp float;
out vec4 result;
void main(){int i=int(gl_FragCoord.x);result=i==0?vec4(.25,-.20,.1,1.):i==1?vec4(0.,0.,0.,1.):vec4(0.);}
`;
export const duckPredictFragment = `#version 300 es
precision highp float;
${duckCommon}
uniform float dt,gravity;
out vec4 result;
void main(){
 vec3 p=duckPosition(),v=texelFetch(duckState,ivec2(2,0),0).xyz,w=texelFetch(duckState,ivec2(3,0),0).xyz;
 vec4 q=duckRotation();v.y-=gravity*dt;
 // Explicit gyroscopic term in body coordinates (Euler's rigid-body equation).
 vec3 wb=qrotate(vec4(-q.xyz,q.w),w);
 w-=qrotate(q,cross(wb,DUCK_INERTIA*wb)/DUCK_INERTIA)*dt;
 // Exact support points of each proxy ellipsoid against the static tank.
 // This remains non-penetrating when the duck rolls; axis-only samples do not.
 for(int shape=0;shape<4;shape++){
  vec3 centre=shape==0?vec3(.025,.055,0.):shape==1?vec3(.17,.24,0.):shape==2?vec3(.19,.39,0.):vec3(.36,.355,0.);
  vec3 radius=shape==0?vec3(.34,.195,.25):shape==1?vec3(.135,.20,.135):shape==2?vec3(.21,.16,.18):vec3(.09,.07,.11);
  for(int side=0;side<5;side++){
   vec3 n=side==0?vec3(0.,1.,0.):side==1?vec3(1.,0.,0.):side==2?vec3(-1.,0.,0.):side==3?vec3(0.,0.,1.):vec3(0.,0.,-1.);
   vec3 direction=qrotate(vec4(-q.xyz,q.w),-n);
   vec3 local=centre+radius*radius*direction/length(radius*direction);
   vec3 r=qrotate(q,local),point=p+r;
   float d=side==0?point.y+.95:side==1?point.x+1.80:side==2?1.80-point.x:side==3?point.z+1.30:1.30-point.z;
   float vn=dot(v+cross(w,r),n);
   if(d+min(vn,0.)*dt<.005){
    float j=max(0.,-vn+max(0.,.005-d)*.18/dt)/(1./DUCK_MASS+dot(n,cross(duckInverseInertia(cross(r,n)),r)));
    v+=n*j/DUCK_MASS;w+=duckInverseInertia(cross(r,n*j));
   }
  }
 }
 // Small air drag; water damping is exchanged with the actual fluid particles.
 v*=exp(-.025*dt);w*=exp(-.04*dt);
 p+=v*dt;q=normalize(q+vec4(cross(w,q.xyz)+q.w*w,-dot(w,q.xyz))*(.5*dt));
 int i=int(gl_FragCoord.x);result=i==0?vec4(p,1.):i==1?q:i==2?vec4(v,0.):vec4(w,0.);
}
`;
export const duckReduceFragment = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D linearSource,angularSource;
uniform float firstLevel;
uniform int count;
layout(location=0) out vec4 linearResult;
layout(location=1) out vec4 angularResult;
void main(){
 ivec2 at=ivec2(gl_FragCoord.xy)*2,size=textureSize(linearSource,0);
 linearResult=vec4(0.);angularResult=vec4(0.);
 for(int y=0;y<2;y++)for(int x=0;x<2;x++){
  ivec2 p=at+ivec2(x,y);
  if(any(greaterThanEqual(p,size))||(firstLevel>.5&&p.x+p.y*256>=count))continue;
  linearResult+=texelFetch(linearSource,p,0);angularResult+=texelFetch(angularSource,p,0);
 }
}
`;
export const duckIntegrateFragment = `#version 300 es
precision highp float;
${duckCommon}
uniform highp sampler2D linearSource,angularSource;
out vec4 result;
void main(){
 int i=int(gl_FragCoord.x);result=texelFetch(duckState,ivec2(i,0),0);
 if(i==2)result.xyz+=texelFetch(linearSource,ivec2(0),0).xyz/DUCK_MASS;
 if(i==3)result.xyz+=duckInverseInertia(texelFetch(angularSource,ivec2(0),0).xyz);
}
`;

/** Ray trace the original Sony/Khronos indexed triangle mesh through a static BVH.
 * Keeping it in the water renderer gives real mesh intersections to refracted and
 * reflected rays, including submerged surfaces, rather than a screen overlay.
 */
export const duckRender = `
${duckCommon}
uniform highp sampler2D duckBVH,duckTriangles,duckAlbedo;
uniform float duckReady;
vec4 meshRead(sampler2D source,int i){return texelFetch(source,ivec2(i%256,i/256),0);}
vec2 meshBox(vec3 o,vec3 invD,vec3 lo,vec3 hi){vec3 a=(lo-o)*invD,b=(hi-o)*invD;vec3 near=min(a,b),far=max(a,b);return vec2(max(near.x,max(near.y,near.z)),min(far.x,min(far.y,far.z)));}
vec4 duckTrace(vec3 ro,vec3 rd,float end){
 vec4 hit=vec4(end,-1.,0.,0.);if(duckReady<.5)return hit;
 vec3 o=duckLocal(ro);vec4 q=duckRotation();vec3 d=qrotate(vec4(-q.xyz,q.w),rd);
 vec3 safe=mix(vec3(1e-8),d,greaterThan(abs(d),vec3(1e-8))),invD=1./safe;
 int stack[24];int top=1;stack[0]=0;
 // Tree depth is 10. The bound exceeds the total 2,047-node imported tree.
 for(int visit=0;visit<2048;visit++){
  if(top==0)break;int node=stack[--top];
  vec4 lo=meshRead(duckBVH,node*2),hi=meshRead(duckBVH,node*2+1);
  vec2 range=meshBox(o,invD,lo.xyz,hi.xyz);
  if(range.y<max(0.,range.x)||range.x>hit.x)continue;
  if(lo.w>=0.){stack[top++]=int(hi.w);stack[top++]=int(lo.w);continue;}
  int start=int(-lo.w-1.);
  for(int j=0;j<6;j++){
   if(j>=int(hi.w))break;int tri=start+j;
   vec3 a=meshRead(duckTriangles,tri*6).xyz,b=meshRead(duckTriangles,tri*6+1).xyz,c=meshRead(duckTriangles,tri*6+2).xyz;
   vec3 e1=b-a,e2=c-a,h=cross(d,e2);float determinant=dot(e1,h);if(abs(determinant)<1e-9)continue;
   vec3 s=o-a;float u=dot(s,h)/determinant;if(u<0.||u>1.)continue;
   vec3 v=cross(s,e1);float w=dot(d,v)/determinant;if(w<0.||u+w>1.)continue;
   float t=dot(e2,v)/determinant;
   if(t>.0004&&t<hit.x)hit=vec4(t,float(tri),u,w);
  }
 }
 return hit;
}
vec3 duckShade(vec4 hit,vec3 rd){
 int tri=int(hit.y)*6;vec3 weight=vec3(1.-hit.z-hit.w,hit.z,hit.w);
 vec4 n0=meshRead(duckTriangles,tri+3),n1=meshRead(duckTriangles,tri+4),n2=meshRead(duckTriangles,tri+5);
 vec3 n=qrotate(duckRotation(),normalize(n0.xyz*weight.x+n1.xyz*weight.y+n2.xyz*weight.z));
 if(dot(n,rd)>0.)n=-n;
 vec2 uv=vec2(dot(vec3(meshRead(duckTriangles,tri).w,meshRead(duckTriangles,tri+1).w,meshRead(duckTriangles,tri+2).w),weight),dot(vec3(n0.w,n1.w,n2.w),weight));
 vec3 albedo=pow(textureLod(duckAlbedo,uv,0.).rgb,vec3(2.2));
 vec3 sun=normalize(vec3(-.6,1.,.35)),halfway=normalize(sun-rd);
 float diffuse=max(0.,dot(n,sun)),spec=pow(max(0.,dot(n,halfway)),85.);
 float facing=pow(1.-max(0.,dot(n,-rd)),5.);
 return albedo*(.34+diffuse*.85*lightPower)+vec3(1.,.96,.82)*spec*lightPower*.6+sky(reflect(rd,n))*(.025+.16*facing);
}
`;
