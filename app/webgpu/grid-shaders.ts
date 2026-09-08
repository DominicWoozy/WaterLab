import { common } from './common.ts';
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
