import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { create, globals } from 'webgpu';
import { WebGPUCaustics } from '../app/webgpu/caustics.ts';
import { RECEIVER_SIZE } from '../app/webgpu/light-space.ts';
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
      const px =
          ((x + 0.5) / CAUSTIC_MAP[0]) * RECEIVER_SIZE[0] -
          RECEIVER_SIZE[0] / 2,
        pz =
          ((z + 0.5) / CAUSTIC_MAP[1]) * RECEIVER_SIZE[1] -
          RECEIVER_SIZE[1] / 2;
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
  16;
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
const empty = await run();
assert.ok(stats(center(empty)).mean > 0.2, 'empty pool retains direct light');
u[29] = 0;
const unobstructed = await run();
assert.ok(stats(center(unobstructed)).std < 0.01, 'empty pool is evenly lit');
fill('wave');
await run();
const drained = await run(0);
assert.deepEqual(
  drained,
  unobstructed,
  'draining removes refraction but preserves direct light',
);

// A suspended parallel slab must refract back into air before the dry floor.
function fieldFrom(fn, top = 3.4) {
  const field = new Float32Array(128 * 160 * 96);
  for (let z = 0; z < 96; z++)
    for (let y = 0; y < 160; y++)
      for (let x = 0; x < 128; x++) {
        const p = [
          -2.08 + (x / 127) * 4.16,
          -1.12 + (y / 159) * 5.2,
          -1.56 + (z / 95) * 3.12,
        ];
        field[x + 128 * (y + 160 * z)] = Math.max(
          0,
          Math.min(8, 1.15 + 40 * fn(...p)),
        );
      }
  device.queue.writeTexture(
    { texture },
    field,
    { bytesPerRow: 512, rowsPerImage: 160 },
    [128, 160, 96],
  );
  device.queue.writeBuffer(bounds, 0, new Float32Array([2 + top, 0, 0, 0]));
}
fieldFrom((_x, y) => Math.min(0.9 - y, y - 0.45), 0.9);
const slab = await run();
let lightRays = new Float32Array(await read(caustics.photons));
assert.equal(lightRays[index + 3], 2, 'suspended slab has entry and exit');
const slabInside = 0.45 / -refracted[1],
  slabAir = (0.9 + 0.97 - 0.45) / -rd[1];
for (const k of [0, 1, 2])
  assert.ok(
    Math.abs(
      lightRays[index + 4 + k] -
        (lightRays[index + k] + refracted[k] * slabInside + rd[k] * slabAir),
    ) < 0.006,
    'slab Snell exit and air landing',
  );
function fresnel(cosine, eta) {
  const ct = Math.sqrt(1 - eta * eta * (1 - cosine * cosine));
  const rs = (eta * cosine - ct) / (eta * cosine + ct),
    rp = (cosine - eta * ct) / (cosine + eta * ct);
  return (rs * rs + rp * rp) / 2;
}
const expectedSlabEnergy =
  (1 - fresnel(-rd[1], 1 / 1.333)) *
  (1 - fresnel(-refracted[1], 1.333)) *
  Math.exp(-0.2 * slabInside);
assert.ok(
  Math.abs(lightRays[index + 9] - expectedSlabEnergy) < 0.004,
  'absorption counts only the submerged distance',
);
assert.ok(
  stats(center(slab)).mean > 0.7,
  'suspended water lights a dry receiver',
);
fieldFrom(
  (_x, y) => Math.max(Math.min(1.5 - y, y - 1.3), Math.min(0.8 - y, y - 0.6)),
  1.5,
);
await run();
lightRays = new Float32Array(await read(caustics.photons));
assert.equal(
  lightRays[index + 3],
  4,
  'stacked sheets must traverse four interfaces',
);

// A high off-center water ball redirects light onto ground OUTSIDE the pool.
fieldFrom((x, y, z) => 0.38 - Math.hypot(x - 1.3, y - 2.1, z), 2.5);
const airborne = await run();
let outsideChanges = 0,
  outsidePeak = 0;
for (let z = 0; z < CAUSTIC_MAP[1]; z++)
  for (let x = 0; x < CAUSTIC_MAP[0]; x++) {
    const px = ((x + 0.5) / CAUSTIC_MAP[0] - 0.5) * RECEIVER_SIZE[0],
      pz = ((z + 0.5) / CAUSTIC_MAP[1] - 0.5) * RECEIVER_SIZE[1];
    if (Math.abs(px) < 2.05 && Math.abs(pz) < 1.55) continue;
    const i = (x + z * CAUSTIC_MAP[0]) * 4 + 1;
    if (Math.abs(airborne[i] - unobstructed[i]) > 0.08) outsideChanges++;
    outsidePeak = Math.max(outsidePeak, airborne[i]);
  }
assert.ok(
  outsideChanges > 100 && outsidePeak > 1.05,
  'flying water produces ground shadows and focused light beyond the tile platform',
);

fieldFrom((x, y) => Math.min(1.3 - y, y + 2 * x), 1.3);
await run();
lightRays = new Float32Array(await read(caustics.photons));
let tir = 0;
for (let i = 0; i < lightRays.length; i += 16) if (lightRays[i + 15] > 0) tir++;
assert.ok(tir > 0, 'steep water exits exercise total internal reflection');

async function duckLight() {
  return new Float32Array(await read(caustics.duckLighting));
}
u[29] = 1;
fill('empty');
await run();
const duckDry = await duckLight();
assert.ok(
  duckDry.some((x) => x > 0.1),
  'duck receives direct light on its own 3D surface',
);
// The imported mesh mirrors material UVs. Lighting must NOT be mirrored.
const mesh = new Float32Array(await read(triangles));
const uvGroups = new Map();
let distinctLightingPairs = 0;
for (let tri = 0; tri < mesh.length / 24; tri++)
  for (let k = 0; k < 3; k++) {
    const at = tri * 24 + k * 4;
    const key = mesh[at + 3].toFixed(4) + ',' + mesh[at + 15].toFixed(4);
    const previous = uvGroups.get(key);
    if (previous !== undefined) {
      const distance = Math.hypot(
        ...[0, 1, 2].map((axis) => mesh[at + axis] - mesh[previous.at + axis]),
      );
      if (
        distance > 0.1 &&
        Math.abs(duckDry[(tri * 3 + k) * 4] - duckDry[previous.vertex * 4]) >
          0.05
      )
        distinctLightingPairs++;
    } else uvGroups.set(key, { at, vertex: tri * 3 + k });
  }
assert.ok(
  distinctLightingPairs > 0,
  'shared material UVs must keep distinct 3D illumination',
);
const dryPhotons = new Uint32Array(await read(caustics.duckDraw))[1];
fieldFrom(
  (x, y, z) =>
    Math.min(1.25 + 0.07 * Math.sin(x * 8) * Math.cos(z * 7) - y, y - 0.85),
  1.4,
);
await run();
const duckWet = await duckLight();
assert.ok(duckWet.every(Number.isFinite), 'duck irradiance remains finite');
let duckChange = 0;
for (let i = 0; i < duckDry.length; i += 4)
  duckChange += Math.abs(duckDry[i] - duckWet[i]);
assert.ok(
  duckChange > 10,
  'suspended refracting water changes lighting on the actual duck surface',
);
u[22] = 0;
await run();
const compareDuck = await duckLight();
for (let i = 0; i < compareDuck.length; i++)
  assert.ok(
    Math.abs(compareDuck[i] - duckDry[i]) < 0.006 * Math.max(1, duckDry[i]),
    'caustic toggle keeps opaque duck light within accumulation rounding',
  );
u[22] = 1;
console.log({
  outsideChanges,
  outsidePeak,
  tir,
  dryPhotons,
  duckChange,
  distinctLightingPairs,
});
assert.deepEqual(errors, []);
console.log({
  filtered,
  flat: flatStats,
  wave: waveStats,
  changed,
  shadowPixels,
});
console.log(
  'PASS: uniform light, multi-interface Snell transport, TIR, airborne ground caustics, 3D duck light/shadow, time invariance, reset/toggles',
);
process.exit(0);
