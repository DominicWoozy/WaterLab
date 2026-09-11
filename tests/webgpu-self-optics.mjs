import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { ComputeKernel, buffer } from '../app/webgpu/compute.ts';
import { selfOpticsCommon } from '../app/webgpu/self-optics-shaders.ts';
Object.assign(globalThis, globals);
globalThis.nativeGPU = create(['backend=metal']);
const adapter = await nativeGPU.requestAdapter();
assert.ok(adapter);
const device = await adapter.requestDevice();
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const source =
  `
struct Scene {right:vec4f,light:vec4f,config:vec4f}
var<private> S:Scene;var<private> mode:u32;
@group(0) @binding(0) var<storage,read_write> output:array<vec4f>;
fn density(p:vec3f)->f32 {
 if(mode==1u || mode==2u){return select(0.,4.,p.y>.3 && p.y<select(.5,.9,mode==2u));}
 if(mode==3u){return select(0.,4.,p.y<0.);}
 return 0.;
}
fn normalAt(p:vec3f)->vec3f {return vec3f(0.,1.,0.);}
fn boxHit(ro:vec3f,rd:vec3f)->vec2f{return vec2f(0.,2.);}
fn duckTrace(ro:vec3f,rd:vec3f,end:f32)->vec4f{return vec4f(end,select(-1.,0.,mode==4u),0.,0.);}
fn duckShade(hit:vec4f,rd:vec3f)->vec3f{return vec3f(1.);}
fn tankContact(p:vec3f,rd:vec3f)->u32{return 0u;}
fn sky(rd:vec3f)->vec3f{return vec3f(1.);}
` +
  selfOpticsCommon +
  `
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 mode=id.x;S=Scene(vec4f(1.),vec4f(1.),vec4f(1.15));
 let light=waterSunVisibility(vec3f(0.),vec3f(0.,1.,0.));
 let hit=reflectedWaterHit(vec3f(0.,.006,0.),vec3f(0.,1.,0.),2.);
 output[2u*id.x]=vec4f(light,hit);
 output[2u*id.x+1u]=vec4f(reflectedWaterHit(vec3f(0.,.006,0.),vec3f(0.,1.,0.),.15));
}`;
const kernel = await ComputeKernel.create(
  device,
  'self-optics fixtures',
  source,
);
const output = buffer(device, 'results', 5 * 32);
const read = device.createBuffer({
  size: output.size,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const e = device.createCommandEncoder(),
  pass = e.beginComputePass();
kernel.dispatch(pass, { 0: output }, 5);
pass.end();
e.copyBufferToBuffer(output, 0, read, 0, output.size);
device.queue.submit([e.finish()]);
await read.mapAsync(GPUMapMode.READ);
const values = new Float32Array(read.getMappedRange().slice(0));
read.unmap();
const rows = Array.from({ length: 5 }, (_, i) =>
  Array.from(values.slice(i * 8, i * 8 + 4)),
);
assert.deepEqual(
  rows[0],
  [1, 1, 1, -1],
  'clear air has no shadow or reflected water',
);
assert.deepEqual(
  rows[3],
  [1, 1, 1, -1],
  'flat water must not intersect or shadow itself',
);
assert.deepEqual(
  rows[4].slice(0, 3),
  [0, 0, 0],
  'opaque blocker occludes sunlight',
);
const sunY = 1 / Math.hypot(0.6, 1, 0.35),
  f = 0.0204 + 0.9796 * (1 - sunY) ** 5;
for (const mode of [1, 2]) {
  const thickness = mode === 1 ? 0.2 : 0.6;
  const expected = [1.25, 0.2, 0.065].map(
    (k) => Math.exp((-k * thickness) / sunY) * (1 - f) ** 2,
  );
  for (let c = 0; c < 3; c++)
    assert.ok(
      Math.abs(rows[mode][c] - expected[c]) < 0.002,
      'colored Beer/Fresnel transmission',
    );
  assert.ok(
    Math.abs(rows[mode][3] - 0.294) < 0.0003,
    'secondary ray locates water surface',
  );
  assert.equal(values[mode * 8 + 4], -1, 'opaque receiver bounds water query');
}
assert.ok(
  rows[2][0] < rows[1][0] && rows[2][0] < rows[2][1],
  'thicker water attenuates more and preserves tint',
);
assert.deepEqual(errors, []);
console.log({ rows, errors });
console.log('PASS: self optics ray and light fixtures');
device.destroy();
process.exit(0);
