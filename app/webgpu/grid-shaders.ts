import { common, NEIGHBOR_CACHE_BASE, recordNeighbor } from './common.ts';
export const gridShaders: Record<string, string> = {
  count:
    common +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read> input:array<Particle>;
 @group(0) @binding(2) var<storage,read_write> counts:array<atomic<u32>>;
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x<P.counts.x){atomicAdd(&counts[key(cell(input[gid.x].pos.xyz))],1u);}
 }`,
  scan:
    common +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read> counts:array<u32>;
 @group(0) @binding(2) var<storage,read_write> starts:array<u32>;
 @group(0) @binding(3) var<storage,read_write> totals:array<u32>;
 var<workgroup> scratch:array<u32,256>;
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
  scratch[lane]=counts[gid.x];workgroupBarrier();
  for(var stride=1u;stride<256u;stride*=2u){
   var v=scratch[lane];if(lane>=stride){v+=scratch[lane-stride];}workgroupBarrier();
   scratch[lane]=v;workgroupBarrier();
  }
  starts[gid.x]=scratch[lane]-counts[gid.x];
  if(lane==255u){totals[group.x]=scratch[lane];}
 }`,
  scanTotals:
    common +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read_write> totals:array<u32>;
 var<workgroup> scratch:array<u32,128>;
 @compute @workgroup_size(128) fn main(@builtin(local_invocation_index) lane:u32){
  var value=0u;if(lane<120u){value=totals[lane];}scratch[lane]=value;workgroupBarrier();
  for(var stride=1u;stride<128u;stride*=2u){var v=scratch[lane];if(lane>=stride){v+=scratch[lane-stride];}workgroupBarrier();scratch[lane]=v;workgroupBarrier();}
  if(lane<120u){totals[lane]=scratch[lane]-value;}
 }`,
  add:
    common +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read_write> starts:array<u32>;
 @group(0) @binding(2) var<storage,read> totals:array<u32>;
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x<30720u){starts[gid.x]+=totals[gid.x/256u];}
  if(gid.x==0u){starts[30720]=P.counts.x;}
 }`,
  scatter:
    common +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read> input:array<Particle>;
 @group(0) @binding(2) var<storage,read_write> output:array<Particle>;
 @group(0) @binding(3) var<storage,read> starts:array<u32>;
 @group(0) @binding(4) var<storage,read_write> cursor:array<atomic<u32>>;
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u){
  let i=gid.x;if(i>=P.counts.x){return;}let bucket=key(cell(input[i].pos.xyz));
  let slot=starts[bucket]+atomicAdd(&cursor[bucket],1u);output[slot]=input[i];
 }`,
};

// Pressure impulses must follow the same permutation as their particle when a
// pressure solve rebuilds the grid. No persistent-ID lookup or fixed capacity.
gridShaders.scatterReactions = gridShaders.scatter
  .replace(
    '@compute @workgroup_size(128) fn main',
    'struct Reaction {linear:vec4f, angular:vec4f}\n@group(0) @binding(5) var<storage,read> impulses:array<Reaction>;\n@group(0) @binding(6) var<storage,read_write> sortedImpulses:array<Reaction>;\n@compute @workgroup_size(128) fn main',
  )
  .replace(
    'output[slot]=input[i];',
    'output[slot]=input[i];sortedImpulses[slot]=impulses[i];',
  );

// Positions remain fixed throughout viscosity and velocity projection. Build an
// exact neighbor cache and pressure factors together, once per physical substep.
gridShaders.prepareVelocity =
  common +
  /* wgsl */ `
 @group(0) @binding(1) var<storage,read> input:array<Particle>;
 @group(0) @binding(2) var<storage,read_write> output:array<Particle>;
 @group(0) @binding(3) var<storage,read_write> starts:array<u32>;
 @group(0) @binding(5) var<storage,read_write> factors:array<vec4f>;
 @group(0) @binding(6) var<storage,read> duck:Duck;
 @group(0) @binding(8) var<storage,read_write> surface:array<vec4f>;
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u){
  let i=gid.x;if(i>=P.counts.x){return;}let p=input[i].pos.xyz;
  let radius=h()+.00001*P.clock.z;let radius2=radius*radius;let size=.225*P.clock.z;
  let lo=cell(p-vec3f(radius));let hi=cell(p+vec3f(radius));var count=0u;
  let wall=wallSupport(p)+duckSupport(p,duck);var rho=wall.w;var sum=0.;var nearby=0.;var grad=wall.xyz;
  for(var z=lo.z;z<=hi.z;z++){
   for(var y=lo.y;y<=hi.y;y++){
    let tile=vec2f(-1.19,-1.53)+vec2f(f32(y),f32(z))*size;
    let gap=max(max(tile-p.yz,p.yz-tile-vec2f(size)),vec2f(0.));
    let remaining=radius2-dot(gap,gap);if(remaining<0.){continue;}
    let extent=sqrt(remaining)+.00001*P.clock.z;
    let x0=clamp(i32(floor((p.x-extent+2.04)/size)),0,31);
    let x1=clamp(i32(floor((p.x+extent+2.04)/size)),0,31);
    let row=32*(y+40*z);let begin=starts[u32(row+x0)];let end=starts[u32(row+x1+1)];
    for(var j=begin;j<end;j++){
     if(j==i){continue;}let diff=p-input[j].pos.xyz;let r2=dot(diff,diff);
     if(r2>=h()*h()||r2<1e-12){continue;}
     ${recordNeighbor()}
     let r=sqrt(r2);let q=1.-r/h();let gradient=(2.*q/(h()*REST*r))*diff;
     rho+=q*q;grad+=gradient;sum+=dot(gradient,gradient);nearby+=1.;
    }
   }
  }
  starts[${NEIGHBOR_CACHE_BASE}u+i]=count;
  surface[i]=vec4f(limited(-h()*grad,2.),(rho+1.)/REST);
  var f=0.;if(nearby>=12.&&rho>REST*.4){f=1./max(sum+dot(grad,grad),1e-6);}
  factors[i]=vec4f(f,rho/REST,nearby,0.);
  var a=input[i];var v=(a.pos.xyz-a.old.xyz)/P.clock.x;
  if(a.old.w<.5){v=vec3f(0.,-1.4,0.);}
  a.vel=vec4f(limited(v,12.)*pow(.998,P.clock.x*60.),a.vel.w);output[i]=a;
 }
`;
