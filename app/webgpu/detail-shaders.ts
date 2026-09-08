// Analytic world-space ellipsoids. These are render-only reconstructions of the
// existing particles: no extra mass, forces, CPU readback, or fixed-size buckets.
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
 if(d.center.w>1.5){
  // The shortest ellipsoid axis is its sheet normal (largest metric column).
  var n=m[0];if(dot(m[1],m[1])>dot(n,n)){n=m[1];}if(dot(m[2],m[2])>dot(n,n)){n=m[2];}n=normalize(n);
  if(dot(n,rd)>0.){n=-n;}
  normal=normalize(mix(normal,n,.88));
 }
 return FOut(vec4f(t,f32(in.index+1u),coverage,d.center.w),vec4f(normal,2.*root),clamp(1.002004-.1002004/(t*dot(rd,S.forward.xyz)),0.,1.));
}
`;

// Accumulate optical thickness across overlapping sheet proxies. Taking only the
// nearest ellipsoid chord would lose most of the sheet's represented water volume.
export const detailThicknessShader =
  detailShader.slice(0, detailShader.indexOf('struct FOut')) +
  /* wgsl */ `
@group(0) @binding(2) var hits:texture_2d<f32>;
@group(0) @binding(3) var normals:texture_2d<f32>;
@fragment fn fragment(in:VOut)->@location(0) f32 {
 let d=details[in.index];if(d.center.w<1.5){discard;}
 let pixel=vec2i(in.position.xy);let nearest=textureLoad(hits,pixel,0);if(nearest.w<1.5){discard;}
 let uv=vec2f(in.position.x/S.view.x,1.-in.position.y/S.view.y);var q=(uv-.5)*vec2f(S.view.x/S.view.y,1.);q.x+=S.view.z;
 let rd=normalize(S.forward.xyz*1.55+S.right.xyz*q.x+S.up.xyz*q.y);let interval=detailRoots(d,S.eye.xyz,rd);
 if(interval.x<.1||abs(interval.x-nearest.x)>.08){discard;}
 let m=detailMetric(d);var n=m[0];if(dot(m[1],m[1])>dot(n,n)){n=m[1];}if(dot(m[2],m[2])>dot(n,n)){n=m[2];}
 if(abs(dot(normalize(n),textureLoad(normals,pixel,0).xyz))<.85){discard;}
 return max(0.,interval.y-interval.x);
}
`;
