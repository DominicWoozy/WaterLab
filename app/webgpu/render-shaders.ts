import { detailCommon } from './detail-shaders.ts';
export function renderShader(filterable: boolean) {
  return (
    detailCommon +
    /* wgsl */ `
struct Scene {eye:vec4f,forward:vec4f,right:vec4f,up:vec4f,view:vec4f,light:vec4f,brush:vec4f,config:vec4f}
struct Duck {pos:vec4f,rotation:vec4f,vel:vec4f,omega:vec4f}
@group(0) @binding(0) var<uniform> S:Scene;
@group(0) @binding(1) var densityVolume:texture_3d<f32>;
@group(0) @binding(2) var densitySampler:sampler;
@group(0) @binding(3) var<storage,read> bounds:array<u32>;
@group(0) @binding(4) var<storage,read> duck:Duck;
@group(0) @binding(5) var<storage,read> bvh:array<vec4f>;
@group(0) @binding(6) var<storage,read> triangles:array<vec4f>;
@group(0) @binding(7) var albedo:texture_2d<f32>;
@group(0) @binding(8) var albedoSampler:sampler;
@group(0) @binding(9) var detailHits:texture_2d<f32>;
@group(0) @binding(10) var detailNormals:texture_2d<f32>;
@group(0) @binding(11) var<storage,read> details:array<Detail>;
const VMIN=vec3f(-2.08,-1.12,-1.56);const VMAX=vec3f(2.08,4.08,1.56);const VSIZE=vec3f(128.,160.,96.);
var<private> volumeTop:f32;
struct VertexOut {@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertex(@builtin(vertex_index) i:u32)->VertexOut {
 let p=vec2f(f32((i<<1u)&2u),f32(i&2u));return VertexOut(vec4f(p*2.-1.,0.,1.),p);
}
fn ray(uv:vec2f)->vec3f {var q=(uv-.5)*vec2f(S.view.x/S.view.y,1.);q.x+=S.view.z;return normalize(S.forward.xyz*1.55+S.right.xyz*q.x+S.up.xyz*q.y);}
fn density(p:vec3f)->f32 {
 if(abs(p.x)>1.86||abs(p.z)>1.36||p.y<-.96||p.y>volumeTop){return 0.;}
 let node=clamp((p-VMIN)/(VMAX-VMIN)*(VSIZE-1.),vec3f(0.),VSIZE-1.);
 ${
   filterable
     ? `return textureSampleLevel(densityVolume,densitySampler,(node+.5)/VSIZE,0.).r;`
     : `
 let base=vec3i(floor(node));let f=fract(node);var sum=0.;
 for(var z=0;z<2;z++){for(var y=0;y<2;y++){for(var x=0;x<2;x++){
 let at=min(base+vec3i(x,y,z),vec3i(127,159,95));let w=select(1.-f,f,vec3i(x,y,z)==vec3i(1));
 sum+=textureLoad(densityVolume,at,0).r*w.x*w.y*w.z;}}}return sum;
 `
 }
}
fn boxHit(ro:vec3f,rd:vec3f)->vec2f {
 let safe=select(vec3f(.00001),rd,abs(rd)>vec3f(.00001));
 let a=(vec3f(-1.87,-.97,-1.37)-ro)/safe;let b=(vec3f(1.87,volumeTop+.01,1.37)-ro)/safe;
 let lo=min(a,b);let hi=max(a,b);return vec2f(max(max(lo.x,lo.y),lo.z),min(min(hi.x,hi.y),hi.z));
}
fn normalAt(p:vec3f)->vec3f {
 let e=vec3f(.022,0.,0.);let n=vec3f(density(p-e.xyy)-density(p+e.xyy),density(p-e.yxy)-density(p+e.yxy),density(p-e.yyx)-density(p+e.yyx));
 if(length(n)>.00001){return normalize(n);}return vec3f(0.,1.,0.);
}
fn qrotate(q:vec4f,v:vec3f)->vec3f {return v+2.*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
fn sky(r:vec3f)->vec3f {
 var c=mix(vec3f(.07,.18,.23),vec3f(.61,.79,.83),smoothstep(-.3,.85,r.y));
 c+=vec3f(.5,.62,.63)*pow(max(dot(r,normalize(vec3f(-.6,.9,.3))),0.),18.);
 c+=vec3f(1.,.91,.72)*pow(max(dot(r,normalize(vec3f(-.6,1.,.35))),0.),260.)*2.5*S.light.x;
 let stripe=smoothstep(.1,.16,r.x)*(1.-smoothstep(.25,.32,r.x))*smoothstep(.15,.35,r.y);
 c+=stripe*vec3f(.85,.97,1.)*.7;c+=.06*sin(r.x*13.+sin(r.z*9.))*max(0.,r.y);return c;
}
fn meshBox(o:vec3f,invD:vec3f,lo:vec3f,hi:vec3f)->vec2f {let a=(lo-o)*invD;let b=(hi-o)*invD;let near=min(a,b);let far=max(a,b);return vec2f(max(near.x,max(near.y,near.z)),min(far.x,min(far.y,far.z)));}
fn duckTrace(ro:vec3f,rd:vec3f,end:f32)->vec4f {
 var hit=vec4f(end,-1.,0.,0.);if(S.config.y<.5){return hit;}
 let invQ=vec4f(-duck.rotation.xyz,duck.rotation.w);let o=qrotate(invQ,ro-duck.pos.xyz);let d=qrotate(invQ,rd);
 let invD=1./select(vec3f(1e-8),d,abs(d)>vec3f(1e-8));var stack:array<u32,24>;var top=1u;stack[0]=0u;
 for(var visit=0;visit<2048;visit++){
  if(top==0u){break;}top--;let node=stack[top];let lo=bvh[node*2u];let hi=bvh[node*2u+1u];let range=meshBox(o,invD,lo.xyz,hi.xyz);
  if(range.y<max(0.,range.x)||range.x>hit.x){continue;}
  if(lo.w>=0.){stack[top]=u32(hi.w);stack[top+1u]=u32(lo.w);top+=2u;continue;}
  let start=u32(-lo.w-1.);
  for(var j=0u;j<6u;j++){
   if(j>=u32(hi.w)){break;}let tri=start+j;let a=triangles[tri*6u].xyz;let b=triangles[tri*6u+1u].xyz;let c=triangles[tri*6u+2u].xyz;
   let e1=b-a;let e2=c-a;let h=cross(d,e2);let det=dot(e1,h);if(abs(det)<1e-9){continue;}
   let s=o-a;let u=dot(s,h)/det;if(u<0.||u>1.){continue;}
   let v=cross(s,e1);let w=dot(d,v)/det;if(w<0.||u+w>1.){continue;}
   let t=dot(e2,v)/det;if(t>.0004&&t<hit.x){hit=vec4f(t,f32(tri),u,w);}
  }
 }
 return hit;
}
fn duckShade(hit:vec4f,rd:vec3f)->vec3f {
 let tri=u32(hit.y)*6u;let w=vec3f(1.-hit.z-hit.w,hit.z,hit.w);let n0=triangles[tri+3u];let n1=triangles[tri+4u];let n2=triangles[tri+5u];
 var n=qrotate(duck.rotation,normalize(n0.xyz*w.x+n1.xyz*w.y+n2.xyz*w.z));if(dot(n,rd)>0.){n=-n;}
 let uv=vec2f(dot(vec3f(triangles[tri].w,triangles[tri+1u].w,triangles[tri+2u].w),w),dot(vec3f(n0.w,n1.w,n2.w),w));
 let color=pow(textureSampleLevel(albedo,albedoSampler,uv,0.).rgb,vec3f(2.2));let sun=normalize(vec3f(-.6,1.,.35));let halfway=normalize(sun-rd);
 let diffuse=max(0.,dot(n,sun));let spec=pow(max(0.,dot(n,halfway)),85.);let facing=pow(1.-max(0.,dot(n,-rd)),5.);
 return color*(.34+diffuse*.85*S.light.x)+vec3f(1.,.96,.82)*spec*S.light.x*.6+sky(reflect(rd,n))*(.025+.16*facing);
}
fn opticalPath(p:vec3f,r:vec3f)->f32 {
 let hit=boxHit(p,r);let end=duckTrace(p,r,max(0.,hit.y)).x;let stepSize=end/72.;var result=0.;
 for(var i=0;i<72;i++){let d=density(p+r*(f32(i)+.5)*stepSize);result+=smoothstep(S.config.x-.18,S.config.x+.18,d)*stepSize;}return result;
}
fn tiles(p:vec2f)->vec3f {let g=abs(fract(p*3.)-.5);let seam=smoothstep(.472,.499,max(g.x,g.y));let c=(floor(p.x*3.)+floor(p.y*3.))-2.*floor((floor(p.x*3.)+floor(p.y*3.))/2.);return mix(mix(vec3f(.65,.68,.66),vec3f(.76,.78,.74),c),vec3f(.12,.24,.26),seam*.75);}
fn room(ro:vec3f,rd:vec3f)->vec3f {
 var c=vec3f(.045,.067,.080);let t=(-1.075-ro.y)/rd.y;
 if(t>0.){let p=ro+rd*t;let d=length(p.xz);c+=vec3f(.018,.026,.028)*exp(-d*.15);let grid=abs(fract(p.xz*.5+.5)-.5);c+=smoothstep(.496,.5,max(grid.x,grid.y))*.014*exp(-d*.18);let shadow=exp(-max(abs(p.x)-1.7,0.)*3.-max(abs(p.z)-1.2,0.)*3.);c*=1.-shadow*.55;}
 let floorT=(-.97-ro.y)/rd.y;if(floorT>0.){let p=ro+rd*floorT;if(abs(p.x)<1.92&&abs(p.z)<1.42){c=tiles(p.xz)*(.43+S.light.x*.28);let rim=max(abs(p.x)/1.92,abs(p.z)/1.42);if(rim>.973){c=vec3f(.24,.35,.37);}}}
 for(var side=0;side<4;side++){
 let axis=select(rd.z,rd.x,side<2);let origin=select(ro.z,ro.x,side<2);let extent=select(1.38,1.88,side<2);let at=select(extent,-extent,side==0||side==2);let hit=(at-origin)/axis;
 if(hit>0.){let p=ro+rd*hit;let lateral=select(abs(p.x),abs(p.z),side<2);let limit=select(1.88,1.38,side<2);
 if(p.y>-.97&&p.y<.8&&lateral<limit){let line=max(exp(-abs(p.y-.8)*180.),exp(-abs(lateral-limit)*180.));c=mix(c,vec3f(.25,.52,.54),.025+line*.45);}}
 }
 return c;
}
fn scene(ro:vec3f,rd:vec3f)->vec3f {let floorT=(-.97-ro.y)/rd.y;let hit=duckTrace(ro,rd,select(1e5,floorT,floorT>0.));if(hit.y>=0.){return duckShade(hit,rd);}return room(ro,rd);}
fn waterColor(p:vec3f,n:vec3f,rd:vec3f,detailId:u32)->vec3f {
  let refracted=refract(rd,n,1./1.333);var exitPoint=p+refracted*.003;var exitRay=refracted;var thickness=0.;
  if(detailId>0u){
   let d=details[detailId-1u];let chord=detailRoots(d,p+refracted*.00001,refracted);let distance=max(0.,chord.y);
   thickness=distance;exitPoint=p+refracted*(distance+.00002);
   {let exitNormal=normalize(detailMetric(d)*(exitPoint-d.center.xyz));let outside=refract(refracted,-exitNormal,1.333);if(dot(outside,outside)>.1){exitRay=normalize(outside);}}
  }
  thickness+=opticalPath(exitPoint,exitRay);let absorb=exp(-vec3f(1.25,.2,.065)*thickness);
  var samplePoint=exitPoint;if(detailId==0u){samplePoint=p+refracted*.006;}
  var transmission=scene(samplePoint,exitRay)*absorb;transmission+=vec3f(.012,.11,.13)*(1.-absorb)*S.light.x*.45;var color=transmission;
  if(S.light.y>.5){let fresnel=.0204+.9796*pow(1.-max(dot(-rd,n),0.),5.);let reflected=reflect(rd,n);let mirrorDuck=duckTrace(p+reflected*.006,reflected,1e5);var reflection=sky(reflected);if(mirrorDuck.y>=0.){reflection=duckShade(mirrorDuck,reflected);}color=mix(transmission,reflection,fresnel);
  let sun=normalize(vec3f(-.6,1.,.35));color+=vec3f(1.,.97,.9)*pow(max(dot(reflect(rd,n),sun),0.),240.)*S.light.x;}
  if(S.light.z>.5){let q=p.xz*10.+n.xz*1.7;let caustic=pow(max(0.,1.-abs(sin(q.x+sin(q.y+S.view.w*.4))+sin(q.y+sin(q.x-S.view.w*.3)))*.7),15.);color+=vec3f(.12,.15,.14)*caustic*S.light.x*min(thickness,.7)*absorb;}
 return color;
}
struct FragmentOut {@location(0) color:vec4f,@builtin(frag_depth) depth:f32}
@fragment fn fragment(in:VertexOut)->FragmentOut {
 volumeTop=bitcast<f32>(bounds[0])-2.+.36;let rd=ray(in.uv);let floorT=(-.97-S.eye.y)/rd.y;
 let primaryDuck=duckTrace(S.eye.xyz,rd,select(1e5,floorT,floorT>0.));var color=room(S.eye.xyz,rd);if(primaryDuck.y>=0.){color=duckShade(primaryDuck,rd);}
 var interval=boxHit(S.eye.xyz,rd);interval.y=min(interval.y,primaryDuck.x);var at=max(0.,interval.x);var last=at;var found=false;
 if(S.light.w<.5&&interval.y>at){for(var i=0;i<384;i++){let d=density(S.eye.xyz+rd*at);if(d>S.config.x){found=true;break;}last=at;at+=select(.026,.014,d>.12);if(at>interval.y){break;}}}
 if(found){
  for(var i=0;i<7;i++){let mid=(at+last)*.5;if(density(S.eye.xyz+rd*mid)>S.config.x){at=mid;}else{last=mid;}}
  let p=S.eye.xyz+rd*at;var n=normalAt(p);if(dot(n,rd)>0.){n=-n;}
  color=waterColor(p,n,rd,0u);
 }
 if(S.eye.w>.5&&S.light.w<.5){
  let pixel=vec2i(in.position.xy);let hit=textureLoad(detailHits,pixel,0);
  if(hit.y>.5&&hit.x<primaryDuck.x&&(!found||hit.x<at)){
   var n=normalize(textureLoad(detailNormals,pixel,0).xyz);
   let p=S.eye.xyz+rd*hit.x;if(dot(n,rd)>0.){n=-n;}
   color=mix(color,waterColor(p,n,rd,u32(hit.y)),hit.z);
  }
 }
 if(S.brush.w>.5){let t=(S.brush.y-S.eye.y)/rd.y;if(t>0.){let p=S.eye.xyz+rd*t;let ring=exp(-pow((length(p.xz-S.brush.xz)-.48)*110.,2.));color+=vec3f(.2,.65,.5)*ring*.55;}}
 color=1.-exp(-color*1.35);let vignette=in.uv-.5;color*=1.-.12*dot(vignette,vignette);
 var depth=1.;if(primaryDuck.y>=0.){depth=clamp(1.002004-.1002004/max(.101,dot(rd*primaryDuck.x,S.forward.xyz)),0.,1.);}
 return FragmentOut(vec4f(pow(color,vec3f(.91)),1.),depth);
}
`
  );
}
export const particlesShader = /* wgsl */ `
struct Scene {eye:vec4f,forward:vec4f,right:vec4f,up:vec4f,view:vec4f,light:vec4f,brush:vec4f,config:vec4f}
struct Particle {pos:vec4f,old:vec4f,vel:vec4f}
@group(0) @binding(0) var<uniform> S:Scene;
@group(0) @binding(1) var<storage,read> particles:array<Particle>;
struct VOut {@builtin(position) position:vec4f,@location(0) local:vec2f,@location(1) depth:f32}
@vertex fn vertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) i:u32)->VOut {
 let corners=array<vec2f,6>(vec2f(-1.,-1.),vec2f(1.,-1.),vec2f(-1.,1.),vec2f(-1.,1.),vec2f(1.,-1.),vec2f(1.,1.));
 let c=corners[vertex];let radius=S.config.z;let p=particles[i].pos.xyz-S.eye.xyz;let z=dot(p,S.forward.xyz);
 let xy=vec2f(dot(p,S.right.xyz),dot(p,S.up.xyz))+c*radius;let aspect=S.view.x/S.view.y;
 return VOut(vec4f(2.*1.55*xy.x/aspect-2.*S.view.z*z/aspect,2.*1.55*xy.y,1.002004*z-.1002004,z),c,z);
}
struct FOut {@location(0) color:vec4f,@builtin(frag_depth) depth:f32}
@fragment fn fragment(in:VOut)->FOut {
 let r2=dot(in.local,in.local);if(r2>1.){discard;}let h=sqrt(1.-r2);let depth=in.depth-h*S.config.z;let n=vec3f(in.local,h);let d=max(0.,dot(n,normalize(vec3f(-.5,.7,1.))));
 return FOut(vec4f(mix(vec3f(.025,.14,.21),vec3f(.32,.85,.78),d),1.),1.002004-.1002004/depth);
}`;
