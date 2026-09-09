// Probe production receiver/water shading directly, without camera screenshots.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../app/webgpu/volume.ts';
import { WebGPURenderer } from '../app/webgpu/renderer.ts';
import { renderShader } from '../app/webgpu/render-shaders.ts';

Object.assign(globalThis, globals);
globalThis.nativeGPU = create(['backend=metal']);
const adapter = await nativeGPU.requestAdapter();
const filtered =
  !process.env.UNFILTERED && adapter.features.has('float32-filterable');
const device = await adapter.requestDevice({
  requiredFeatures: filtered ? ['float32-filterable'] : [],
});
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const sim = await WebGPUSimulation.create(device);
const volume = await WebGPUVolume.create(device);
const renderer = await WebGPURenderer.create(device, 'rgba8unorm');
const bvh = await readFile('public/models/duck/bvh.bin');
const triangles = await readFile('public/models/duck/triangles.bin');
const bytes = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
renderer.setModel(bytes(bvh), bytes(triangles));
device.queue.writeBuffer(
  sim.duck,
  0,
  new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
);
const setupTarget = device.createTexture({
  size: [64, 8],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
let encoder = device.createCommandEncoder();
renderer.encode(
  encoder,
  setupTarget.createView(),
  sim,
  volume,
  { eye: [0, 2, 3], forward: [0, -1, 0], right: [1, 0, 0], up: [0, 0, -1] },
  64,
  8,
  { light: 1.3, reflection: true, caustics: true, particles: false },
);
// Use controlled irradiance to isolate receiver compositing; ray-map generation
// is checked independently in webgpu-caustic-map.mjs.
encoder = device.createCommandEncoder();
const mapPass = encoder.beginRenderPass({
  colorAttachments: [
    {
      view: renderer.caustics.view,
      loadOp: 'clear',
      storeOp: 'store',
      clearValue: [2, 2, 2, 1],
    },
  ],
});
mapPass.end();
device.queue.submit([encoder.finish()]);
const prefix = renderShader(filtered).split('struct FragmentOut')[0];
const code =
  prefix +
  /* wgsl */ `
@fragment fn fragment(in:VertexOut)->@location(0) vec4f {
 volumeTop=.8;let row=u32(in.position.y);let x=(in.position.x/64.-.5)*1.6;
 let floor=vec3f(x,-.97,.1);let down=vec3f(0.,-1.,0.);
 if(row==0u){return vec4f(room(floor+vec3f(0.,1.,0.),down),1.);}
 if(row==1u){return vec4f(room(floor+vec3f(.3,1.,.1),normalize(vec3f(-.3,-1.,-.1))),1.);}
 if(row==2u){return vec4f(room(vec3f(x,.3,.1),-down),1.);}
 if(row==3u){return vec4f(waterColor(vec3f(x,0.,.1),-down,down,0u),1.);}
 if(row==4u){return vec4f(opticalPath(vec3f(x,-.003,.1),down),0.,0.,1.);}
 if(row==5u){return vec4f(scene(vec3f(0.,2.,0.),down),duckTrace(vec3f(0.,2.,0.),down,2.97).y);}
 if(row==6u){return vec4f(waterColor(vec3f(1.75,0.,0.),-down,normalize(vec3f(1.,-.1,0.)),0u),1.);}
 return vec4f(floorLighting(floor),1.);
}`;
const shaderModule = device.createShaderModule({ code });
const messages = await shaderModule.getCompilationInfo();
assert.deepEqual(
  messages.messages.filter((x) => x.type === 'error'),
  [],
);
const pipeline = await device.createRenderPipelineAsync({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [renderer.surfaceLayout],
  }),
  vertex: { module: shaderModule, entryPoint: 'vertex' },
  fragment: {
    module: shaderModule,
    entryPoint: 'fragment',
    targets: [{ format: 'rgba32float' }],
  },
});
const target = device.createTexture({
  size: [64, 8],
  format: 'rgba32float',
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const kept = (globalThis.readbacks = []);
const field = new Float32Array(128 * 160 * 96);
for (let z = 0; z < 96; z++)
  for (let y = 0; y < 160; y++) {
    const height = -1.12 + (y / 159) * 5.2;
    if (height < 0) field.fill(8, 128 * (y + z * 160), 128 * (y + z * 160 + 1));
  }
function fill(wet) {
  device.queue.writeTexture(
    { texture: volume.texture },
    wet ? field : new Float32Array(field.length),
    { bytesPerRow: 512, rowsPerImage: 160 },
    [128, 160, 96],
  );
}
async function probe(
  on,
  { light = 1.3, reflection = true, time = 1.7, particles = false } = {},
) {
  const u = new Float32Array(32);
  u.set([64, 8, 0, time], 16);
  u.set([light, +reflection, +on, +particles], 20);
  // Keep the real duck mesh active; the transmission oracle uses rays clear of it.
  u.set([1.15, 1, 0.021, Math.cbrt(0.2)], 28);
  device.queue.writeBuffer(renderer.uniform, 0, u);
  const staging = device.createBuffer({
    size: 64 * 8 * 16,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      { view: target.createView(), loadOp: 'clear', storeOp: 'store' },
    ],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, renderer.surfaceGroups.get(sim.duck));
  pass.draw(3);
  pass.end();
  encoder.copyTextureToBuffer(
    { texture: target },
    { buffer: staging, bytesPerRow: 1024 },
    [64, 8],
  );
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const range = staging.getMappedRange();
  kept.push([staging, range]);
  const values = new Float32Array(range.slice(0));
  staging.unmap();
  assert.ok(values.every(Number.isFinite));
  return values;
}
function rowDiff(a, b, row) {
  return Array.from(
    { length: 64 * 3 },
    (_, i) =>
      a[(row * 64 + Math.floor(i / 3)) * 4 + (i % 3)] -
      b[(row * 64 + Math.floor(i / 3)) * 4 + (i % 3)],
  );
}
fill(true);
const off = await probe(false),
  on = await probe(true);
assert.ok(
  Math.max(...rowDiff(on, off, 0)) > 0.04,
  'wet floor must receive visible light',
);
for (let x = 0; x < 64; x++)
  for (let k = 0; k < 3; k++)
    assert.ok(
      Math.abs(on[x * 4 + k] - on[(64 + x) * 4 + k]) < 2e-5,
      'same floor point must keep its pattern across camera directions',
    );
assert.ok(
  Math.max(...rowDiff(on, off, 2).map(Math.abs)) < 1e-7,
  'no floor intersection means no caustic',
);
assert.ok(on[5 * 64 * 4 + 3] >= 0, 'duck probe must hit the actual mesh');
assert.ok(
  Math.max(...rowDiff(on, off, 5).map(Math.abs)) < 1e-7,
  'foreground duck must not receive a screen overlay',
);
assert.ok(
  Math.max(...rowDiff(on, off, 6).map(Math.abs)) < 1e-7,
  'water ray missing the pool floor must not glow',
);
// Use x<-.45 to keep these vertical water rays clear of the duck hull.
for (let x = 0; x < 12; x++)
  for (let k = 0; k < 3; k++) {
    const floorDelta = on[x * 4 + k] - off[x * 4 + k];
    const thickness = on[(4 * 64 + x) * 4];
    const expected =
      floorDelta * Math.exp(-[1.25, 0.2, 0.065][k] * thickness) * (1 - 0.0204);
    const actual = on[(3 * 64 + x) * 4 + k] - off[(3 * 64 + x) * 4 + k];
    assert.ok(
      Math.abs(actual - expected) < 2e-5,
      'caustic must be transmitted, absorbed and Fresnel weighted like the floor',
    );
  }
assert.deepEqual(await probe(true), on, 'paused state must be deterministic');
const later = await probe(true, { time: 2.7 });
assert.deepEqual(later, on, 'time alone must not animate a fixed caustic map');
for (const setting of [{ light: 0 }, { particles: true }]) {
  const a = await probe(false, setting),
    b = await probe(true, setting);
  assert.deepEqual(a, b, 'disabled lighting/debug must suppress caustics');
}
fill(false);
assert.deepEqual(
  await probe(true),
  await probe(false),
  'dry pool must not retain a caustic pattern',
);
assert.deepEqual(errors, []);
console.log(
  `PASS (${filtered ? 'filtered' : 'manual trilinear'}): floor coordinates, dry mask, occlusion, no surface emission, absorption/Fresnel, toggles and pause`,
);
process.exit(0);
