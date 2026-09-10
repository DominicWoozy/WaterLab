import { common } from './common.ts';
const bindings = /* wgsl */ `
@group(0) @binding(1) var<storage,read> input:Duck;
@group(0) @binding(2) var<storage,read_write> output:Duck;
struct Reaction {linear:vec4f,angular:vec4f}
@group(0) @binding(3) var<storage,read> reaction:array<Reaction>;
`;
export const duckShaders: Record<string, string> = {
  duckInitialize:
    common +
    bindings +
    /* wgsl */ `
 @compute @workgroup_size(1) fn main(){output=Duck(vec4f(.25,-.2,.1,1.),vec4f(0.,0.,0.,1.),vec4f(0.),vec4f(0.));}`,
  duckPredict:
    common +
    bindings +
    /* wgsl */ `
 @compute @workgroup_size(1) fn main(){
 var p=input.pos.xyz;var v=input.vel.xyz;var w=input.omega.xyz;var q=input.rotation;let dt=P.clock.x;v.y-=P.clock.w*dt;
 let wb=qrotate(conjugate(q),w);w-=qrotate(q,cross(wb,INERTIA*wb)/INERTIA)*dt;
 let centres=array<vec3f,4>(vec3f(.025,.055,0.),vec3f(.17,.24,0.),vec3f(.19,.39,0.),vec3f(.36,.355,0.));
 let radii=array<vec3f,4>(vec3f(.34,.195,.25),vec3f(.135,.20,.135),vec3f(.21,.16,.18),vec3f(.09,.07,.11));
 let normals=array<vec3f,5>(vec3f(0.,1.,0.),vec3f(1.,0.,0.),vec3f(-1.,0.,0.),vec3f(0.,0.,1.),vec3f(0.,0.,-1.));
 for(var shape=0;shape<4;shape++){let centre=centres[shape];let radius=radii[shape];
 for(var side=0;side<5;side++){let n=normals[side];let direction=qrotate(conjugate(q),-n);
 let local=centre+radius*radius*direction/length(radius*direction);let r=qrotate(q,local);let point=p+r;
 var d=point.y+.95;if(side==1){d=point.x+1.8;}if(side==2){d=1.8-point.x;}if(side==3){d=point.z+1.3;}if(side==4){d=1.3-point.z;}
 let vn=dot(v+cross(w,r),n);
 if(d+min(vn,0.)*dt<.005){let j=max(0.,-vn+max(0.,.005-d)*.18/dt)/(1./MASS+dot(n,cross(inverseInertia(cross(r,n),q),r)));
 v+=n*j/MASS;w+=inverseInertia(cross(r,n*j),q);}
 }}
 v*=exp(-.025*dt);w*=exp(-.04*dt);p+=v*dt;q=normalize(q+vec4f(cross(w,q.xyz)+q.w*w,-dot(w,q.xyz))*(.5*dt));
 output=Duck(vec4f(p,1.),q,vec4f(v,0.),vec4f(w,0.));
 }`,
  duckIntegrate:
    common +
    bindings +
    /* wgsl */ `
 @compute @workgroup_size(1) fn main(){var d=input;d.vel+=vec4f(reaction[0].linear.xyz/MASS,0.);d.omega+=vec4f(inverseInertia(reaction[0].angular.xyz,d.rotation),0.);output=d;}`,
};
for (const [name, limit] of [
  ['reduceParticles', 'P.counts.x'],
  ['reduceGroups', '(P.counts.x+255u)/256u'],
] as const) {
  duckShaders[name] =
    common +
    /* wgsl */ `
 struct Reaction {linear:vec4f,angular:vec4f}
 @group(0) @binding(1) var<storage,read> input:array<Reaction>;
 @group(0) @binding(2) var<storage,read_write> output:array<Reaction>;
 var<workgroup> linear:array<vec4f,256>;var<workgroup> angular:array<vec4f,256>;
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 var a=Reaction(vec4f(0.),vec4f(0.));
 ${name === 'reduceGroups' ? `for(var j=lane;j<${limit};j+=256u){a.linear+=input[j].linear;a.angular+=input[j].angular;}` : `if(gid.x<${limit}){a=input[gid.x];}`}
 linear[lane]=a.linear;angular[lane]=a.angular;workgroupBarrier();
 for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){linear[lane]+=linear[lane+stride];angular[lane]+=angular[lane+stride];}workgroupBarrier();}
 if(lane==0u){output[group.x]=Reaction(linear[0],angular[0]);}
 }`;
}
