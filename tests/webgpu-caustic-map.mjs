import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { create, globals } from 'webgpu';
import { WebGPUCaustics } from '../app/webgpu/caustics.ts';
import { CAUSTIC_GRID, CAUSTIC_MAP } from '../app/webgpu/caustic-shaders.ts';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const adapter = await gpu.requestAdapter();
const filtered =
  !process.env.UNFILTERED && adapter.features.has('float32-filterable');
const device = await adapter.requestDevice({
  requiredFeatures: filtered ? ['float32-filterable'] : [],
});
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
function buf(size) {
  return device.createBuffer({
    size,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
}
const uniform = device.createBuffer({
  size: 128,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
const caustics = await WebGPUCaustics.create(device, uniform);
const texture = device.createTexture({
  size: [128, 160, 96],
  dimension: '3d',
  format: 'r32float',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
const bounds = buf(16),
  duck = buf(64);
let bvh = buf(16),
  triangles = buf(16);
const sampler = device.createSampler({
  minFilter: filtered ? 'linear' : 'nearest',
  magFilter: filtered ? 'linear' : 'nearest',
});
const volume = { view: texture.createView(), bounds };
const u = new Float32Array(32);
u.set([1.3, 1, 1, 0], 20);
u.set([1.15, 0, 0.021, Math.cbrt(0.2)], 28);
device.queue.writeBuffer(
  duck,
  0,
  new Float32Array([0, -0.2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
);
const kept = (globalThis.readbacks = []);
async function read(buffer) {
  const stage = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(buffer, 0, stage, 0, buffer.size);
  device.queue.submit([e.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const range = stage.getMappedRange();
  kept.push([stage, range]);
  const copy = range.slice(0);
  stage.unmap();
  return copy;
}
function half(v) {
  const sign = v & 32768 ? -1 : 1,
    exp = (v >> 10) & 31,
    frac = v & 1023;
  return (
    sign *
    (exp === 0
      ? frac * 2 ** -24
      : exp === 31
        ? frac
          ? NaN
          : Infinity
        : (1 + frac / 1024) * 2 ** (exp - 15))
  );
}
async function run(count = 50000) {
  device.queue.writeBuffer(uniform, 0, u);
  const e = device.createCommandEncoder();
  caustics.encode(e, volume, sampler, duck, bvh, triangles, count);
  const staging = buf(CAUSTIC_MAP[0] * CAUSTIC_MAP[1] * 8);
  e.copyTextureToBuffer(
    { texture: caustics.texture },
    { buffer: staging, bytesPerRow: CAUSTIC_MAP[0] * 8 },
    [...CAUSTIC_MAP],
  );
  device.queue.submit([e.finish()]);
  const data = Float32Array.from(new Uint16Array(await read(staging)), half);
  assert.ok(data.every(Number.isFinite));
  return data;
}
function fill(mode = 'flat', phase = 0) {
  const field = new Float32Array(128 * 160 * 96);
  for (let z = 0; z < 96; z++)
    for (let y = 0; y < 160; y++)
      for (let x = 0; x < 128; x++) {
        const px = -2.08 + (x / 127) * 4.16,
          py = -1.12 + (y / 159) * 5.2,
          pz = -1.56 + (z / 95) * 3.12;
        const height =
          -0.35 +
          (mode === 'wave'
            ? 0.06 * Math.sin(px * 8 + phase) * Math.cos(pz * 7 - phase)
            : 0);
        field[x + 128 * (y + 160 * z)] =
          mode === 'empty'
            ? 0
            : Math.max(0, Math.min(8, 1.15 + (height - py) * 40));
      }
  device.queue.writeTexture(
    { texture },
    field,
    { bytesPerRow: 512, rowsPerImage: 160 },
    [128, 160, 96],
  );
  device.queue.writeBuffer(
    bounds,
    0,
    new Float32Array([mode === 'empty' ? 0 : 2 - 0.35 + 0.06, 0, 0, 0]),
  );
}
function center(data) {
  const a = [];
  for (let z = 0; z < CAUSTIC_MAP[1]; z++)
    for (let x = 0; x < CAUSTIC_MAP[0]; x++) {
      const px = ((x + 0.5) / CAUSTIC_MAP[0]) * 3.84 - 1.92,
        pz = ((z + 0.5) / CAUSTIC_MAP[1]) * 2.84 - 1.42;
      if (Math.abs(px) < 0.65 && Math.abs(pz) < 0.65)
        a.push(data[(x + z * CAUSTIC_MAP[0]) * 4 + 1]);
    }
  return a;
}
function stats(a) {
  const mean = a.reduce((s, x) => s + x, 0) / a.length;
  return {
    mean,
    min: Math.min(...a),
    max: Math.max(...a),
    std: Math.sqrt(a.reduce((s, x) => s + (x - mean) ** 2, 0) / a.length),
  };
}
fill();
const flat = await run(),
  flatStats = stats(center(flat));
assert.ok(
  flatStats.mean > 0.65 && flatStats.mean < 1,
  'flat surface irradiance includes transmission and absorption',
);
assert.ok(
  flatStats.std / flatStats.mean < 0.015,
  'flat water must not invent a caustic pattern',
);
const photons = new Float32Array(await read(caustics.photons));
const index =
  (Math.floor(CAUSTIC_GRID[0] / 2) +
    CAUSTIC_GRID[0] * Math.floor(CAUSTIC_GRID[1] / 2)) *
  12;
assert.equal(photons[index + 7], 1);
const sun = [-0.6, 1, 0.35],
  length = Math.hypot(...sun),
  rd = sun.map((x) => -x / length),
  eta = 1 / 1.333;
const refracted = [
  rd[0] * eta,
  -Math.sqrt(1 - eta * eta * (rd[0] ** 2 + rd[2] ** 2)),
  rd[2] * eta,
];
const distance = (-0.97 - photons[index + 1]) / refracted[1];
for (const k of [0, 1, 2])
  assert.ok(
    Math.abs(
      photons[index + 4 + k] - (photons[index + k] + refracted[k] * distance),
    ) < 0.004,
    'Snell refraction must hit the analytic floor point',
  );
u[19] = 9;
assert.deepEqual(
  await run(),
  flat,
  'changing time without moving water must not animate light',
);
fill('wave');
const wave = await run(),
  waveStats = stats(center(wave));
assert.ok(
  waveStats.std > 0.12 && waveStats.max > flatStats.mean * 1.4,
  'curved water must concentrate light',
);
fill('wave', 0.8);
const moved = await run();
const a = center(wave),
  b = center(moved);
const changed = a.reduce((s, x, i) => s + Math.abs(x - b[i]), 0) / a.length;
assert.ok(changed > 0.07, 'moving the actual surface must move its caustics');
// Real mesh occludes light before or after entering water.
fill();
for (const [name, file] of [
  ['bvh', 'bvh.bin'],
  ['triangles', 'triangles.bin'],
]) {
  const bytes = await readFile('public/models/duck/' + file);
  const buffer = buf(bytes.byteLength);
  device.queue.writeBuffer(buffer, 0, bytes);
  if (name === 'bvh') bvh = buffer;
  else triangles = buffer;
}
u[29] = 1;
caustics.clearCache();
const shadow = await run();
let shadowPixels = 0;
for (let i = 1; i < flat.length; i += 4)
  if (flat[i] > 0.5 && shadow[i] < flat[i] * 0.2) shadowPixels++;
assert.ok(shadowPixels > 100, 'duck must cast a caustic shadow');
fill('empty');
assert.ok(
  (await run()).every((x) => x === 0),
  'no water must produce no light map',
);
fill('wave');
await run();
assert.ok(
  (await run(0)).every((x) => x === 0),
  'draining must clear the old map',
);
assert.deepEqual(errors, []);
console.log({
  filtered,
  flat: flatStats,
  wave: waveStats,
  changed,
  shadowPixels,
});
console.log(
  'PASS: flat uniform irradiance, Snell landing, actual wave response, time invariance, duck shadow, empty/reset',
);
process.exit(0);
