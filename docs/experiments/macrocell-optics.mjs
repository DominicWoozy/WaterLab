// Conservative cache coverage + paired production optical paths. No browser QA.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../../app/webgpu/volume.ts';
import { WebGPURenderer } from '../../app/webgpu/renderer.ts';
import { renderShader } from '../../app/webgpu/render-shaders.ts';
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
const volume = await WebGPUVolume.create(device),
  sim = await WebGPUSimulation.create(device);
const renderer = await WebGPURenderer.create(device, 'rgba8unorm');
const w = 128,
  h = 9;
const setup = device.createTexture({
  size: [w, h],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
renderer.encode(
  device.createCommandEncoder(),
  setup.createView(),
  sim,
  volume,
  { eye: [0, 2, 3], forward: [0, -1, 0], right: [1, 0, 0], up: [0, 0, -1] },
  w,
  h,
  { light: 1.3, reflection: true, caustics: false, particles: false },
);
const u = new Float32Array(32);
u.set([w, h, 0, 0], 16);
u.set([1.3, 1, 0, 0], 20);
u.set([1.15, 0, 0.021, 0.58], 28);
device.queue.writeBuffer(renderer.uniform, 0, u);
const code = `
@fragment fn fragment(in:VertexOut)->@location(0) vec4f {
 volumeTop=2.;let group=u32(in.position.y)/3u;let row=u32(in.position.y)%3u;
 let angle=(in.position.x+.37)*2.39996323;let y=sin(in.position.x*1.743+.23);
 let rd=normalize(vec3f(cos(angle),y,sin(angle)));
 var ro=vec3f(sin(in.position.x*3.11)*1.4,-.4,cos(in.position.x*.79)*1.);
 if(group==1u){ro=vec3f(-.8,.65,0.);}
 if(group==2u){ro=vec3f(.6,1.2,0.);}
 let path=traceTransmission(ro,rd,true,0u);
 if(row==0u){return vec4f(path.direction,path.distance);}
 if(row==1u){return vec4f(path.origin,path.weight);}
 return vec4f(f32(path.events),f32(path.complete),density(ro),1.);
}`;
const pipelines = await Promise.all(
  [false, true].map(async (enabled) => {
    const module = device.createShaderModule({
      code:
        renderShader(filtered, enabled).split('struct FragmentOut')[0] + code,
    });
    const info = await module.getCompilationInfo();
    assert.deepEqual(
      info.messages.filter((x) => x.type === 'error'),
      [],
    );
    return device.createRenderPipelineAsync({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [renderer.surfaceLayout],
      }),
      vertex: { module, entryPoint: 'vertex' },
      fragment: {
        module,
        entryPoint: 'fragment',
        targets: [{ format: 'rgba32float' }],
      },
    });
  }),
);
const target = device.createTexture({
  size: [w, h],
  format: 'rgba32float',
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const kept = (globalThis.kept = []);
async function readBuffer(buffer) {
  const staging = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
  device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const range = staging.getMappedRange();
  kept.push([staging, range]);
  const out = new Float32Array(range.slice(0));
  staging.unmap();
  return out;
}
async function fill(field) {
  device.queue.writeBuffer(volume.temp, 0, field);
  device.queue.writeTexture(
    { texture: volume.texture },
    field,
    { bytesPerRow: 512, rowsPerImage: 160 },
    [128, 160, 96],
  );
  const e = device.createCommandEncoder();
  volume.encodeRanges(e);
  device.queue.submit([e.finish()]);
  await device.queue.onSubmittedWorkDone();
}
// Single hot samples on shared macrocell faces, edges and corners. Scatter each
// node into the independently enumerated set of macros whose interpolation uses it.
const impulse = new Float32Array(128 * 160 * 96);
for (const [x, y, z, v] of [
  [64, 40, 48, 7],
  [127, 159, 95, 13],
  [0, 0, 0, 3],
  [4, 4, 4, 11],
  [65, 41, 49, 5],
])
  impulse[x + 128 * (y + 160 * z)] = v;
await fill(impulse);
let ranges = await readBuffer(volume.ranges);
for (const [x, y, z, v] of [
  [64, 40, 48, 7],
  [127, 159, 95, 13],
  [0, 0, 0, 3],
  [4, 4, 4, 11],
  [65, 41, 49, 5],
]) {
  const cells = (p, max) =>
    [Math.floor(p / 4), ...(p > 0 && p % 4 === 0 ? [p / 4 - 1] : [])].filter(
      (c) => c <= max,
    );
  for (const bz of cells(z, 23))
    for (const by of cells(y, 39))
      for (const bx of cells(x, 31)) {
        const i = 2 * (1 + bx + 32 * (by + 40 * bz));
        assert.ok(ranges[i] <= 0);
        assert.ok(
          ranges[i + 1] >= v,
          'all shared support nodes must be covered',
        );
      }
}
const field = new Float32Array(impulse.length);
for (let z = 0; z < 96; z++)
  for (let y = 0; y < 160; y++)
    for (let x = 0; x < 128; x++) {
      const px = -2.08 + (x / 127) * 4.16,
        py = -1.12 + (y / 159) * 5.2,
        pz = -1.56 + (z / 95) * 3.12;
      const water = py < 0.15 + 0.06 * Math.sin(px * 4) * Math.cos(pz * 3);
      const slab =
        Math.abs(py - 0.65) < 0.07 &&
        px > -1.4 &&
        px < -0.2 &&
        Math.abs(pz) < 0.8;
      const sphere = (px - 0.6) ** 2 + (py - 1.2) ** 2 + pz * pz < 0.22 ** 2;
      field[x + 128 * (y + 160 * z)] = water || slab || sphere ? 4 : 0;
    }
await fill(field);
ranges = await readBuffer(volume.ranges);
assert.equal(ranges[0], 1);
// Independent interior interpolation probes including block borders.
for (let i = 0; i < 10000; i++) {
  const x = ((i * 0.61803398875) % 1) * 126.999,
    y = ((i * 0.41421356) % 1) * 158.999,
    z = ((i * 0.7320508) % 1) * 94.999;
  let d = 0;
  for (let dz = 0; dz < 2; dz++)
    for (let dy = 0; dy < 2; dy++)
      for (let dx = 0; dx < 2; dx++)
        d +=
          field[
            Math.floor(x) +
              dx +
              128 * (Math.floor(y) + dy + 160 * (Math.floor(z) + dz))
          ] *
          (dx ? x % 1 : 1 - (x % 1)) *
          (dy ? y % 1 : 1 - (y % 1)) *
          (dz ? z % 1 : 1 - (z % 1));
  const offset =
    2 *
    (1 + Math.floor(x / 4) + 32 * (Math.floor(y / 4) + 40 * Math.floor(z / 4)));
  assert.ok(
    d >= ranges[offset] && d <= ranges[offset + 1],
    'trilinear samples stay in cached bounds',
  );
}
async function probe(pipeline) {
  const e = device.createCommandEncoder();
  const pass = e.beginRenderPass({
    colorAttachments: [
      { view: target.createView(), loadOp: 'clear', storeOp: 'store' },
    ],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, renderer.surfaceGroups.get(sim.duck));
  pass.draw(3);
  pass.end();
  const staging = device.createBuffer({
    size: w * h * 16,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  e.copyTextureToBuffer(
    { texture: target },
    { buffer: staging, bytesPerRow: w * 16 },
    [w, h],
  );
  device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const range = staging.getMappedRange();
  kept.push([staging, range]);
  const result = new Float32Array(range.slice(0));
  staging.unmap();
  return result;
}
const reference = await probe(pipelines[0]),
  accelerated = await probe(pipelines[1]);
let maxError = 0,
  events = 0;
for (let i = 0; i < reference.length; i++) {
  assert.ok(Number.isFinite(accelerated[i]));
  maxError = Math.max(maxError, Math.abs(reference[i] - accelerated[i]));
}
for (let group = 0; group < 3; group++)
  for (let x = 0; x < w; x++) {
    const o = ((group * 3 + 2) * w + x) * 4;
    assert.ok(reference[o + 2] > 1.15, 'fixture ray starts inside water');
    assert.equal(accelerated[o], reference[o], 'same interface event count');
    assert.equal(
      accelerated[o + 1],
      reference[o + 1],
      'same completion/budget state',
    );
    events += reference[o];
  }
assert.ok(events > 100, 'fixtures exercise many interfaces');
assert.ok(
  maxError < 0.003,
  `optical path changes exceed tolerance: ${maxError}`,
);
// Compare the complete primary + refracted image on a frozen field as well.
const height = 96;
const norm = (a) => {
  const n = Math.hypot(...a);
  return a.map((x) => x / n);
};
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const eye = [3, 2.5, 4],
  forward = norm(eye.map((x) => -x)),
  right = norm(cross(forward, [0, 1, 0])),
  up = cross(right, forward);
device.queue.writeBuffer(volume.bounds, 0, new Float32Array([3.64, 0, 0, 0]));
const scratch = device.createTexture({
  size: [w, height],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const init = device.createCommandEncoder();
renderer.encode(
  init,
  scratch.createView(),
  sim,
  volume,
  { eye, forward, right, up },
  w,
  height,
  { light: 1.3, reflection: true, caustics: false, particles: false },
);
device.queue.submit([init.finish()]);
const sceneTarget = device.createTexture({
  size: [w, height],
  format: 'rgba32float',
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const depth = device.createTexture({
  size: [w, height],
  format: 'depth32float',
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const images = [];
for (const enabled of [false, true]) {
  const module = device.createShaderModule({
    code: renderShader(filtered, enabled),
  });
  const pipeline = await device.createRenderPipelineAsync({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [renderer.surfaceLayout],
    }),
    vertex: { module, entryPoint: 'vertex' },
    fragment: {
      module,
      entryPoint: 'fragment',
      targets: [{ format: 'rgba32float' }],
    },
    depthStencil: {
      format: 'depth32float',
      depthWriteEnabled: true,
      depthCompare: 'always',
    },
  });
  const e = device.createCommandEncoder(),
    pass = e.beginRenderPass({
      colorAttachments: [
        { view: sceneTarget.createView(), loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: {
        view: depth.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1,
      },
    });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, renderer.surfaceGroups.get(sim.duck));
  pass.draw(3);
  pass.end();
  const read = device.createBuffer({
    size: w * height * 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  e.copyTextureToBuffer(
    { texture: sceneTarget },
    { buffer: read, bytesPerRow: w * 16 },
    [w, height],
  );
  device.queue.submit([e.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const range = read.getMappedRange();
  kept.push([read, range]);
  images.push(new Float32Array(range.slice(0)));
  read.unmap();
}
let imageMax = 0,
  imageSquared = 0;
for (let i = 0; i < images[0].length; i++) {
  const delta = Math.abs(images[0][i] - images[1][i]);
  assert.ok(Number.isFinite(delta));
  imageMax = Math.max(imageMax, delta);
  imageSquared += delta * delta;
}
assert.ok(imageMax < 0.003, `primary image differs: ${imageMax}`);
await fill(new Float32Array(field.length));
ranges = await readBuffer(volume.ranges);
assert.ok(
  ranges.slice(2).every((v) => Math.abs(v) < 0.00002),
  'empty/reset field invalidates old occupied ranges',
);
assert.deepEqual(errors, []);
console.log(
  JSON.stringify({
    filtered,
    rays: w * 3,
    maxError,
    events,
    rangeBytes: volume.ranges.size,
    imageMax,
    imageRms: Math.sqrt(imageSquared / images[0].length),
    errors,
  }),
);
console.log(
  'PASS: conservative support coverage, trilinear bounds, optical paths, logical budgets and reset',
);
process.exit(0);
