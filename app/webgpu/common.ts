export const CAPACITY = 50000;
export const GRID_CELLS = 32 * 40 * 24;
export const WORKGROUP = 128;

export const common = /* wgsl */ `
struct Particle { pos: vec4f, old: vec4f, vel: vec4f }
struct Params {
  counts: vec4u, // count, quality, previousCount, reset reaction
  clock: vec4f, // dt, time, particleScale, gravity
  forces: vec4f, // viscosity, agitation, shake, surface tension mode
  brush: vec4f,
  motion: vec4f, // brush vx/vz, pour x/z
  splash: vec4f,
}
struct Duck { pos: vec4f, rotation: vec4f, vel: vec4f, omega: vec4f }
@group(0) @binding(0) var<uniform> P: Params;
const REST = 3.6;
const MASS = .026;
const INERTIA = vec3f(.00105,.00075,.00145);
fn h()->f32 {return .17*P.clock.z;}
fn particleMass()->f32 {return (2.*3.14159265/15.)*h()*h()*h()/REST;}
fn limited(v:vec3f,m:f32)->vec3f {return v*min(1.,m/max(length(v),.000001));}
fn bound(p:vec3f)->vec3f {return clamp(p,vec3f(-1.78,-.917,-1.28),vec3f(1.78,3.8,1.28));}
fn cell(p:vec3f)->vec3i {return clamp(vec3i(floor((p-vec3f(-2.04,-1.19,-1.53))/(.225*P.clock.z))),vec3i(0),vec3i(31,39,23));}
fn key(c:vec3i)->u32 {return u32(c.x+32*(c.y+40*c.z));}
fn qrotate(q:vec4f,v:vec3f)->vec3f {return v+2.*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
fn conjugate(q:vec4f)->vec4f {return vec4f(-q.xyz,q.w);}
fn wallSupport(p:vec3f)->vec4f {
 let dlo=p-vec3f(-1.78,-.917,-1.28);let dhi=vec3f(1.78,3.8,1.28)-p;
 var solid=0.;var gradient=vec3f(0.);
 for(var side=0;side<6;side++){
  let axis=side%3;let lower=side<3;let a=clamp(select(dhi[axis],dlo[axis],lower)/h(),0.,1.);
  let a2=a*a;let a3=a2*a;let a4=a3*a;let a5=a4*a;
  let cap=.5-1.25*a+2.5*a3-2.5*a4+.75*a5;
  let slope=(1.25-7.5*a2+10.*a3-3.75*a4)/h();var n=vec3f(0.);n[axis]=select(-1.,1.,lower);
  gradient=gradient*(1.-cap)+n*slope*(1.-solid);solid+=cap*(1.-solid);
 }
 return vec4f(gradient,solid*REST);
}
fn ellipsoid(p:vec3f,r:vec3f)->f32 {return (length(p/r)-1.)*min(r.x,min(r.y,r.z));}
fn duckHull(p:vec3f,d: Duck)->vec4f {
 let x=qrotate(conjugate(d.rotation),p-d.pos.xyz);
 var r=vec3f(.34,.195,.25);var v=x-vec3f(.025,.055,0.);var best=ellipsoid(v,r);
 var nr=vec3f(.135,.20,.135);var nd=x-vec3f(.17,.24,0.);var n=ellipsoid(nd,nr);
 if(n<best){best=n;v=nd;r=nr;}
 nr=vec3f(.21,.16,.18);nd=x-vec3f(.19,.39,0.);n=ellipsoid(nd,nr);
 if(n<best){best=n;v=nd;r=nr;}
 return vec4f(qrotate(d.rotation,normalize(v/(r*r)+vec3f(1e-8))),best);
}
fn duckSupport(p:vec3f,d:Duck)->vec4f {
 let hull=duckHull(p,d);let a=clamp(hull.w/h(),0.,1.);
 let a2=a*a;let a3=a2*a;let a4=a3*a;let a5=a4*a;
 let cap=.5-1.25*a+2.5*a3-2.5*a4+.75*a5;
 let slope=(1.25-7.5*a2+10.*a3-3.75*a4)/h();return vec4f(hull.xyz*slope,cap*REST);
}
fn duckVelocity(p:vec3f,d:Duck)->vec3f {return d.vel.xyz+cross(d.omega.xyz,p-d.pos.xyz);}
fn inverseInertia(t:vec3f,q:vec4f)->vec3f {return qrotate(q,qrotate(conjugate(q),t)/INERTIA);}
`;

// A complete range per cell, with no fixed bucket/neighbor capacity. Adjacent X
// cells form one contiguous interval. Rebuild after the second pressure correction;
// at each neighbor query, at most one .042*scale pressure/contact shift is outstanding.
export function neighbors(body: string) {
  return /* wgsl */ `
 let base=cell(p);let reach=1;
 for(var dz=-reach;dz<=reach;dz++){let z=base.z+dz;if(z<0||z>=24){continue;}
  for(var dy=-reach;dy<=reach;dy++){let y=base.y+dy;if(y<0||y>=40){continue;}
   let x0=max(0,base.x-reach);let x1=min(31,base.x+reach);let row=32*(y+40*z);
   let start=starts[u32(row+x0)];let end=starts[u32(row+x1+1)];
   for(var j=start;j<end;j++){
    if(j==i){continue;}let diff=p-input[j].pos.xyz;let r2=dot(diff,diff);
    if(r2>=h()*h()||r2<1e-12){continue;}
    let r=sqrt(r2);let q=1.-r/h();let gradient=(2.*q/(h()*REST*r))*diff;
    ${body}
   }
  }
 }
 `;
}
