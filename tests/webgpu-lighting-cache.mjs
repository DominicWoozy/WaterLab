// Frozen receiver transport must remain identical when only presentation changes.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../app/webgpu/volume.ts';
import { WebGPURenderer } from '../app/webgpu/renderer.ts';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const adapter = await gpu.requestAdapter();
const features =
  !process.env.UNFILTERED && adapter.features.has('float32-filterable')
    ? ['float32-filterable']
    : [];
if (adapter.features.has('timestamp-query')) features.push('timestamp-query');
const device = await adapter.requestDevice({ requiredFeatures: features });
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const sim = await WebGPUSimulation.create(device),
  volume = await WebGPUVolume.create(device);
const renderer = await WebGPURenderer.create(device, 'rgba8unorm');
const bytes = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const model = [
  bytes(await readFile('public/models/duck/bvh.bin')),
  bytes(await readFile('public/models/duck/triangles.bin')),
];
renderer.setModel(...model);
device.queue.writeBuffer(
  sim.duck,
  0,
  new Float32Array([0, -0.25, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
);
const field = new Float32Array(128 * 160 * 96);
for (let z = 0; z < 96; z++)
  for (let y = 0; y < 160; y++)
    for (let x = 0; x < 128; x++) {
      const px = -2.08 + (x / 127) * 4.16,
        py = -1.12 + (y / 159) * 5.2,
        pz = -1.56 + (z / 95) * 3.12;
      const height = -0.35 + 0.1 * Math.sin(px * 4) * Math.cos(pz * 3);
      field[x + 128 * (y + 160 * z)] = Math.max(
        0,
        Math.min(4, 1.15 + (height - py) * 25),
      );
    }
device.queue.writeTexture(
  { texture: volume.texture },
  field,
  { bytesPerRow: 512, rowsPerImage: 160 },
  [128, 160, 96],
);
device.queue.writeBuffer(volume.bounds, 0, new Float32Array([2, 0, 0, 0]));
const w = 640,
  h = 400;
const target = device.createTexture({
  size: [w, h],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const norm = (a) => {
  const n = Math.hypot(...a);
  return a.map((x) => x / n);
};
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
function camera(eye) {
  const forward = norm(eye.map((x) => -x)),
    right = norm(cross(forward, [0, 1, 0]));
  return { eye, forward, right, up: cross(right, forward) };
}
let view = camera([3, 2.5, 4]),
  options = {
    light: 1.3,
    reflection: true,
    caustics: true,
    particles: false,
    lightingRevision: 1,
  };
let solves = 0;
const solve = renderer.caustics.encode.bind(renderer.caustics);
renderer.caustics.encode = (...args) => {
  solves++;
  return solve(...args);
};
const timing = device.features.has('timestamp-query')
  ? device.createQuerySet({ type: 'timestamp', count: 240 })
  : null;
let frame = 0;
async function render(read = false, timed = false) {
  const encoder = device.createCommandEncoder();
  let started = false;
  const e =
    timed && timing
      ? new Proxy(encoder, {
          get(target, key) {
            if (key === 'beginComputePass' || key === 'beginRenderPass')
              return (descriptor) => {
                const stamps = { querySet: timing };
                if (!started) {
                  stamps.beginningOfPassWriteIndex = frame * 2;
                  started = true;
                }
                if (
                  key === 'beginRenderPass' &&
                  descriptor?.label === 'water and duck'
                )
                  stamps.endOfPassWriteIndex = frame * 2 + 1;
                return target[key]({
                  ...descriptor,
                  ...(Object.keys(stamps).length > 1
                    ? { timestampWrites: stamps }
                    : {}),
                });
              };
            const value = Reflect.get(target, key, target);
            return typeof value === 'function' ? value.bind(target) : value;
          },
        })
      : encoder;
  renderer.encode(e, target.createView(), sim, volume, view, w, h, options);
  let staging;
  if (read) {
    staging = device.createBuffer({
      size: w * h * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    e.copyTextureToBuffer(
      { texture: target },
      { buffer: staging, bytesPerRow: w * 4 },
      [w, h],
    );
  }
  device.queue.submit([e.finish()]);
  if (!read) return;
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}
async function equivalent(label, shouldRecompute) {
  const before = solves,
    actual = await render(true);
  assert.equal(solves - before, +shouldRecompute, label + ' invalidation');
  const revision = options.lightingRevision;
  delete options.lightingRevision;
  const expected = await render(true);
  options.lightingRevision = revision;
  let max = 0,
    total = 0;
  for (let i = 0; i < actual.length; i++) {
    const d = Math.abs(actual[i] - expected[i]);
    max = Math.max(max, d);
    total += d;
  }
  // Photon atomic ordering can differ in the last framebuffer quantization bit.
  assert.ok(max <= 1, `${label}: stale image, max delta ${max}`);
  console.log(
    JSON.stringify({
      label,
      maxByteDelta: max,
      meanByteDelta: total / actual.length,
    }),
  );
  await render();
  await device.queue.onSubmittedWorkDone();
}
await equivalent('first frame', true);
await equivalent('unchanged scene', false);
view = camera([-3, 1.7, 4]);
await equivalent('camera only', false);
options.brush = [0.4, -0.25, 0.1, 1];
await equivalent('brush overlay', false);
options.reflection = false;
await equivalent('reflection toggle', false);
options.light = 0.8;
await equivalent('light intensity', true);
options.caustics = false;
await equivalent('caustics off', true);
options.caustics = true;
await equivalent('caustics on', true);
device.queue.writeBuffer(
  sim.duck,
  0,
  new Float32Array([0.5, -0.25, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
);
options.lightingRevision++;
await equivalent('changed duck pose', true);
field.fill(0);
device.queue.writeTexture(
  { texture: volume.texture },
  field,
  { bytesPerRow: 512, rowsPerImage: 160 },
  [128, 160, 96],
);
options.lightingRevision++;
await equivalent('drained scene', true);
renderer.setModel(...model);
await equivalent('model reload', true);
// Restore water and benchmark presentation-only frames. Every moving physical
// frame still recomputes; this is not a claim about whole-simulation speed.
for (let z = 0; z < 96; z++)
  for (let y = 0; y < 160; y++)
    for (let x = 0; x < 128; x++)
      field[x + 128 * (y + 160 * z)] = Math.max(
        0,
        Math.min(
          4,
          1.15 +
            (-0.35 +
              0.1 * Math.sin(x * 0.13) * Math.cos(z * 0.12) -
              (-1.12 + (y / 159) * 5.2)) *
              25,
        ),
      );
device.queue.writeTexture(
  { texture: volume.texture },
  field,
  { bytesPerRow: 512, rowsPerImage: 160 },
  [128, 160, 96],
);
options = { light: 1.3, reflection: true, caustics: true, particles: false };
const modes = [],
  wall = { cached: [], fresh: [] };
const initialSolves = solves;
for (frame = 0; frame < 120; frame++) {
  const cached = Math.floor(frame / 4) % 2 === 0;
  // First frame in a cached block populates the cache; omit it in timing stats.
  if (cached) options.lightingRevision = 99;
  else delete options.lightingRevision;
  modes.push(cached ? 'cached' : 'fresh');
  const start = performance.now();
  await render(false, true);
  await device.queue.onSubmittedWorkDone();
  if (frame >= 16 && frame % 4 !== 0)
    wall[cached ? 'cached' : 'fresh'].push(performance.now() - start);
}
assert.equal(
  solves - initialSolves,
  75,
  '15 populated cache blocks plus 60 fresh frames',
);
console.log(
  JSON.stringify({
    submission_and_completion_ms: Object.fromEntries(
      Object.entries(wall).map(([k, v]) => [
        k,
        v.reduce((a, b) => a + b, 0) / v.length,
      ]),
    ),
  }),
);
if (timing) {
  const resolve = device.createBuffer({
    size: 1920,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const read = device.createBuffer({
    size: 1920,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const e = device.createCommandEncoder();
  e.resolveQuerySet(timing, 0, 240, resolve, 0);
  e.copyBufferToBuffer(resolve, 0, read, 0, 1920);
  device.queue.submit([e.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const timestamps = new BigUint64Array(read.getMappedRange().slice(0));
  read.unmap();
  const rows = { cached: [], fresh: [] };
  for (let f = 16; f < 120; f++)
    if (f % 4 !== 0)
      rows[modes[f]].push(
        Number(timestamps[f * 2 + 1] - timestamps[f * 2]) / 1e6,
      );
  console.log(
    JSON.stringify({
      presentation_gpu_ms: Object.fromEntries(
        Object.entries(rows).map(([k, v]) => [
          k,
          v.reduce((a, b) => a + b, 0) / v.length,
        ]),
      ),
      resolution: [w, h],
    }),
  );
}
assert.deepEqual(errors, []);
console.log('PASS lighting invalidation and cached vs freshly traced images');
process.exit(0);
