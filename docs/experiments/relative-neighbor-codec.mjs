// Experimental codec fixture: apply relative-neighbor-cache.patch first.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import {
  common,
  CAPACITY,
  GRID_STORAGE_WORDS,
  NEIGHBOR_CACHE_BASE,
  recordNeighbor,
  finishNeighborCache,
  cachedNeighbors,
} from '../../app/webgpu/common.ts';
import { ComputeKernel, buffer } from '../../app/webgpu/compute.ts';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const device = await (await gpu.requestAdapter()).requestDevice();
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const lists = [
  [],
  [66000],
  [66000, 66001],
  [66000, 66002, 69999],
  Array.from({ length: 95 }, (_, i) => 66000 + i),
  Array.from({ length: 96 }, (_, i) => 66000 + i),
  [0, 65535],
  [0, 65536],
  [0, 69999],
  Array.from({ length: 97 }, (_, i) => i),
];
const n = lists.length;
const metadata = buffer(device, 'list bounds', n * 16),
  ids = buffer(device, 'source ids', n * 100 * 4),
  starts = buffer(device, 'cache', GRID_STORAGE_WORDS * 4),
  params = buffer(
    device,
    'params',
    96,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  ),
  out = buffer(device, 'decoded ids', n * 100 * 4),
  dummy = buffer(device, 'unused particles', 48);
const caseData = new Uint32Array(n * 4),
  data = new Uint32Array(n * 100);
for (let i = 0; i < n; i++) {
  caseData.set(
    [lists[i].length, lists[i][0] ?? 0, (lists[i].at(-1) ?? 0) + 1, 0],
    i * 4,
  );
  data.set(lists[i], i * 100);
}
device.queue.writeBuffer(metadata, 0, caseData);
device.queue.writeBuffer(ids, 0, data);
const encoderShader =
  common +
  `
@group(0) @binding(1) var<storage,read> caseData:array<vec4u>;
@group(0) @binding(2) var<storage,read> ids:array<u32>;
@group(0) @binding(3) var<storage,read_write> starts:array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;var count=0u;var cachePacked=0u;let cacheOrigin=caseData[i].y;let cacheWide=caseData[i].z-cacheOrigin>65536u;
 starts[${NEIGHBOR_CACHE_BASE + CAPACITY}u+i]=cacheOrigin;
 for(var k=0u;k<caseData[i].x;k++){let j=ids[i*100u+k];${recordNeighbor()}}
 ${finishNeighborCache()}
}`;
const decodeShader =
  common +
  `
@group(0) @binding(1) var<storage,read> input:array<Particle>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;
@group(0) @binding(3) var<storage,read> starts:array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if((starts[${NEIGHBOR_CACHE_BASE}u+i]&0x7fffffffu)>96u){return;}
 let p=vec3f(0.);var cursor=0u;
 ${cachedNeighbors('output[i*100u+cursor]=j;cursor++;').replaceAll('if(r2>=h()*h()||r2<1e-12){continue;}', '')}
}`;
const write = await ComputeKernel.create(device, 'encode', encoderShader),
  read = await ComputeKernel.create(device, 'decode', decodeShader);
async function run() {
  const e = device.createCommandEncoder();
  const pass = e.beginComputePass();
  write.dispatch(pass, { 0: params, 1: metadata, 2: ids, 3: starts }, n);
  read.dispatch(pass, { 0: params, 1: dummy, 2: out, 3: starts }, n);
  pass.end();
  device.queue.submit([e.finish()]);
}
const kept = (globalThis.kept = []);
async function download(b) {
  const s = buffer(
    device,
    'readback',
    b.size,
    GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  );
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(b, 0, s, 0, b.size);
  device.queue.submit([e.finish()]);
  await s.mapAsync(GPUMapMode.READ);
  const r = s.getMappedRange();
  kept.push([s, r]);
  const a = new Uint32Array(r.slice(0));
  s.unmap();
  return a;
}
await run();
let result = await download(out),
  cache = await download(starts);
for (let i = 0; i < n; i++) {
  const header = cache[NEIGHBOR_CACHE_BASE + i];
  assert.equal(header & 0x7fffffff, lists[i].length);
  assert.equal(
    header >>> 31,
    caseData[i * 4 + 2] - caseData[i * 4 + 1] > 65536 ? 1 : 0,
  );
  if (lists[i].length <= 96)
    assert.deepEqual(
      Array.from(result.slice(i * 100, i * 100 + lists[i].length)),
      lists[i],
    );
}
// Reuse a previously wide list as a short packed odd-length list.
caseData.set([3, 66000, 66003, 0], 8 * 4);
data.set([66000, 66001, 66002], 800);
device.queue.writeBuffer(metadata, 0, caseData);
device.queue.writeBuffer(ids, 0, data);
await run();
result = await download(out);
cache = await download(starts);
assert.equal(cache[NEIGHBOR_CACHE_BASE + 8], 3);
assert.deepEqual(Array.from(result.slice(800, 803)), [66000, 66001, 66002]);
assert.deepEqual(errors, []);
console.log(
  'PASS: exact GPU roundtrip, IDs above 65535, short-offset boundary, u32 fallback, odd/even/empty/96/97 counts, wide-to-packed reuse',
);
process.exit(0);
