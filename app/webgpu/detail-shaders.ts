// Analytic world-space ellipsoids. These are render-only reconstructions of the
// primary particles. center.w=1, m2.w=coverage; no extra particles or sheet patches.
export const detailCommon = /* wgsl */ `
struct Detail { center:vec4f, m0:vec4f, m1:vec4f, m2:vec4f }
fn detailMetric(d:Detail)->mat3x3f{return mat3x3f(d.m0.xyz,d.m1.xyz,d.m2.xyz);}
fn detailRoots(d:Detail,ro:vec3f,rd:vec3f)->vec2f {
 let m=detailMetric(d);let o=ro-d.center.xyz;let a=dot(rd,m*rd);let b=dot(o,m*rd);let closest=o-rd*(b/a);let delta=1.-dot(closest,m*closest);
 if(delta<0.){return vec2f(-1.);}let root=sqrt(delta/a);return vec2f(-b/a-root,-b/a+root);
}
`;
export const detailShader =
  detailCommon +
  /* wgsl */ `
struct Scene {eye:vec4f,forward:vec4f,right:vec4f,up:vec4f,view:vec4f,light:vec4f,brush:vec4f,config:vec4f}
@group(0) @binding(0) var<uniform> S:Scene;
@group(0) @binding(1) var<storage,read> details:array<Detail>;
struct VOut {@builtin(position) position:vec4f,@location(0) @interpolate(flat) index:u32}
@vertex fn vertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) i:u32)->VOut {
 let corners=array<vec2f,6>(vec2f(-1.,-1.),vec2f(1.,-1.),vec2f(-1.,1.),vec2f(-1.,1.),vec2f(1.,-1.),vec2f(1.,1.));
 let d=details[i];let p=d.center.xyz-S.eye.xyz;let z=dot(p,S.forward.xyz);let xy=vec2f(dot(p,S.right.xyz),dot(p,S.up.xyz));
 let radius=d.m0.w;let padding=z/(1.55*S.view.y);
 let extent=radius*(1.+length(xy)/max(z,.1))/max(.1,1.-radius/max(z,.1))+padding;
 let q=xy+corners[vertex]*extent;let aspect=S.view.x/S.view.y;
 return VOut(vec4f(2.*1.55*q.x/aspect-2.*S.view.z*z/aspect,2.*1.55*q.y,1.002004*z-.1002004,z),i);
}
struct FOut {@location(0) hit:vec4f,@location(1) normal:vec4f,@builtin(frag_depth) depth:f32}
@fragment fn fragment(in:VOut)->FOut {
 let uv=vec2f(in.position.x/S.view.x,1.-in.position.y/S.view.y);
 var q=(uv-.5)*vec2f(S.view.x/S.view.y,1.);q.x+=S.view.z;
 let rd=normalize(S.forward.xyz*1.55+S.right.xyz*q.x+S.up.xyz*q.y);
 let d=details[in.index];let m=detailMetric(d);let o=S.eye.xyz-d.center.xyz;
 let a=dot(rd,m*rd);let b=dot(o,m*rd);let closest=o-rd*(b/a);let delta=1.-dot(closest,m*closest);
 // Analytic coverage at silhouettes, including subpixel drops. The geometry
 // radius stays physical; only the bounding quad includes a one-pixel margin.
 let width=max(fwidth(delta),1e-7);let coverage=smoothstep(-width*.5,width*.5,delta);
 if(coverage<.01){discard;}
 let root=sqrt(max(0.,delta)/a);let t=-b/a-root;if(t<=.1){discard;}
 let p=S.eye.xyz+rd*t;var normal=normalize(m*(p-d.center.xyz));
 return FOut(vec4f(t,f32(in.index+1u),coverage*d.m2.w,d.center.w),vec4f(normal,2.*root),clamp(1.002004-.1002004/(t*dot(rd,S.forward.xyz)),0.,1.));
}
`;
