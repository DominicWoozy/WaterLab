import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../app/webgpu/volume.ts';
import { WebGPURenderer } from '../app/webgpu/renderer.ts';
import { writeFile } from 'node:fs/promises';
import { deflateSync } from 'node:zlib';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredFeatures: adapter.features.has('float32-filterable')
    ? ['float32-filterable']
    : [],
});
const errors = [];
let lost;
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
device.lost.then((info) => (lost = info));
const kept = (globalThis.readbacks = []);
async function read(b) {
  const dst = device.createBuffer({
    size: b.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let e = device.createCommandEncoder();
  e.copyBufferToBuffer(b, 0, dst, 0, b.size);
  device.queue.submit([e.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const range = dst.getMappedRange();
  kept.push([dst, range]);
  const out = range.slice(0);
  dst.unmap();
  return out;
}
const sim = await WebGPUSimulation.create(device),
  volume = await WebGPUVolume.create(device),
  renderer = await WebGPURenderer.create(device, 'rgba8unorm');
let e = device.createCommandEncoder();
sim.reset(e, 50000);
device.queue.submit([e.finish()]);
const positions = [];
// A free-standing, single-particle-layer film with an actual hole, not a filled disk.
for (let z = -15; z <= 15; z++)
  for (let x = -15; x <= 15; x++) {
    if (Math.hypot(x, z) < 4) continue;
    positions.push([
      x * 0.045,
      0.25 + 0.07 * Math.sin(x * 0.045 * 4) * Math.cos(z * 0.045 * 3),
      z * 0.045,
    ]);
  }
const drops = [
  [-0.7, 1.1, 0],
  [-0.35, 1.1, 0],
  [0, 1.1, 0],
  [0.35, 1.1, 0],
  [0.7, 1.1, 0],
];
positions.push(...drops);
const data = new Float32Array(positions.length * 12);
positions.forEach((p, i) => {
  data.set([...p, 1, ...p, 1, 0, 0, 0, i], i * 12);
});
sim.count = positions.length;
device.queue.writeBuffer(sim.state, 0, data);
const w = 1024,
  h = 768;
const target = device.createTexture({
  size: [w, h],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const norm = (a) => {
    const l = Math.hypot(...a);
    return a.map((v) => v / l);
  },
  cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
const eye = [0, 2.3, 4],
  forward = norm([0, 0.4 - eye[1], -eye[2]]),
  right = norm(cross(forward, [0, 1, 0])),
  up = cross(right, forward),
  camera = { eye, forward, right, up };
function pixel(p) {
  const d = p.map((v, i) => v - eye[i]),
    dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0),
    z = dot(d, forward);
  return [
    Math.floor(w / 2 + h * ((1.55 * dot(d, right)) / z - 0.19)),
    Math.floor(h / 2 - (h * 1.55 * dot(d, up)) / z),
  ];
}
async function pixels() {
  const dst = device.createBuffer({
    size: w * h * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  let e = device.createCommandEncoder();
  e.copyTextureToBuffer(
    { texture: target },
    { buffer: dst, bytesPerRow: w * 4 },
    [w, h],
  );
  device.queue.submit([e.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const range = dst.getMappedRange();
  kept.push([dst, range]);
  const out = new Uint8Array(range.slice(0));
  dst.unmap();
  return out;
}
function crc(buf) {
  let c = 0xffffffff;
  for (const v of buf) {
    c ^= v;
    for (let i = 0; i < 8; i++) c = (c >>> 1) ^ (c & 1 ? 0xedb88320 : 0);
  }
  return (c ^ 0xffffffff) >>> 0;
}
async function png(path, pixels) {
  const chunk = (s, b) => {
    const n = Buffer.from(s),
      l = Buffer.alloc(4),
      c = Buffer.alloc(4);
    l.writeUInt32BE(b.length);
    c.writeUInt32BE(crc(Buffer.concat([n, b])));
    return Buffer.concat([l, n, b, c]);
  };
  const hdr = Buffer.alloc(13);
  hdr.writeUInt32BE(w);
  hdr.writeUInt32BE(h, 4);
  hdr[8] = 8;
  hdr[9] = 6;
  const rows = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++)
    rows.set(pixels.subarray(y * w * 4, (y + 1) * w * 4), y * (w * 4 + 1) + 1);
  await writeFile(
    path,
    Buffer.concat([
      Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
      chunk('IHDR', hdr),
      chunk('IDAT', deflateSync(rows)),
      chunk('IEND', Buffer.alloc(0)),
    ]),
  );
}
let images = [];
for (const enabled of [false, true]) {
  e = device.createCommandEncoder();
  volume.encode(e, sim, enabled);
  device.queue.submit([e.finish()]);
  e = device.createCommandEncoder();
  renderer.encode(e, target.createView(), sim, volume, camera, w, h, {
    light: 1.3,
    reflection: true,
    caustics: true,
    particles: false,
  });
  device.queue.submit([e.finish()]);
  await device.queue.onSubmittedWorkDone();
  images.push(await pixels());
  const counts = new Uint32Array(await read(volume.detailDraw));
  if (!enabled) {
    assert.equal(counts[1], 0);
    continue;
  }
  assert.equal(counts[0], 6);
  assert.ok(
    counts[1] > positions.length * 0.8 && counts[1] <= positions.length,
  );
  const features = new Float32Array(await read(volume.details)),
    state = new Float32Array(await read(sim.state));
  let droplets = 0,
    sheets = 0;
  const mass = (((2 * Math.PI) / 15) * (0.17 * Math.cbrt(0.2)) ** 3) / 3.6;
  for (let i = 0; i < counts[1]; i++) {
    const d = features.subarray(i * 16, i * 16 + 16);
    assert.ok(d.every(Number.isFinite));
    if (d[3] === 1) droplets++;
    if (d[3] === 2) sheets++;
    const j = d[11];
    assert.ok(j < sim.count);
    for (let a = 0; a < 3; a++) assert.equal(d[a], state[j * 12 + a]);
    const m = [d[4], d[8], d[12], d[5], d[9], d[13], d[6], d[10], d[14]];
    const det =
      m[0] * (m[4] * m[8] - m[5] * m[7]) -
      m[1] * (m[3] * m[8] - m[5] * m[6]) +
      m[2] * (m[3] * m[7] - m[4] * m[6]);
    const v = (4 * Math.PI) / 3 / Math.sqrt(det);
    assert.ok(Math.abs(v / mass - 1) < 0.002);
  }
  assert.equal(droplets, drops.length);
  assert.ok(sheets > 700);
  // No change to physics positions (compare by persistent identity through GPU reorder).
  for (let i = 0; i < sim.count; i++) {
    const id = state[i * 12 + 11];
    for (let a = 0; a < 3; a++)
      assert.equal(state[i * 12 + a], data[id * 12 + a]);
  }
  // Inspect the actual detail render target, not just whether the final scene has color.
  const tex = renderer.detailHits,
    dst = device.createBuffer({
      size: w * h * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  e = device.createCommandEncoder();
  e.copyTextureToBuffer(
    { texture: tex },
    { buffer: dst, bytesPerRow: w * 16 },
    [w, h],
  );
  device.queue.submit([e.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const range = dst.getMappedRange();
  kept.push([dst, range]);
  const hits = new Float32Array(range.slice(0));
  dst.unmap();
  assert.ok(hits.every(Number.isFinite));
  const hole = pixel([0, 0.25, 0]);
  assert.equal(
    hits[(hole[0] + w * hole[1]) * 4 + 1],
    0,
    'Actual sheet hole remains empty',
  );
  let measured = [];
  for (const p of drops) {
    const [cx, cy] = pixel(p);
    let area = 0;
    for (let y = cy - 12; y <= cy + 12; y++)
      for (let x = cx - 12; x <= cx + 12; x++) {
        const at = (x + w * y) * 4;
        if (hits[at + 3] === 1) area += hits[at + 2];
      }
    assert.ok(
      area > 80 && area < 400,
      'Analytic drop has a small but visible footprint',
    );
    measured.push(area);
  }
  const sample = pixel([0.3, 0.25 + 0.07 * Math.sin(1.2), 0]);
  const optical = device.createBuffer({
    size: 768,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  e = device.createCommandEncoder();
  for (const [texture, p, offset] of [
    [renderer.sheetThickness, hole, 0],
    [renderer.sheetThickness, sample, 256],
    [renderer.detailNormals, sample, 512],
  ])
    e.copyTextureToBuffer(
      { texture, origin: [...p, 0] },
      { buffer: optical, offset, bytesPerRow: 256 },
      [1, 1],
    );
  device.queue.submit([e.finish()]);
  await optical.mapAsync(GPUMapMode.READ);
  const opticalRange = optical.getMappedRange();
  kept.push([optical, opticalRange]);
  const half = new Uint16Array(opticalRange.slice(0));
  optical.unmap();
  const f16 = (u) => {
    const e = (u >>> 10) & 31;
    const m = u & 1023;
    return (
      (u & 32768 ? -1 : 1) *
      (e === 0
        ? m * 2 ** -24
        : e === 31
          ? Infinity
          : (1 + m / 1024) * 2 ** (e - 15))
    );
  };
  const totalThickness = f16(half[128]),
    nearestChord = f16(half[259]);
  assert.equal(f16(half[0]), 0);
  assert.ok(totalThickness > nearestChord * 1.5);
  const normal = [f16(half[256]), f16(half[257]), f16(half[258])];
  const qx = (sample[0] + 0.5 - w / 2) / h + 0.19,
    qy = (h / 2 - sample[1] - 0.5) / h;
  const ray = norm(
    forward.map((v, i) => v * 1.55 + right[i] * qx + up[i] * qy),
  );
  const nearT = hits[(sample[0] + sample[1] * w) * 4];
  let expectedThickness = 0;
  const dot = (a, b) => a.reduce((v, x, i) => v + x * b[i], 0);
  for (let i = 0; i < counts[1]; i++) {
    const d = features.subarray(i * 16, i * 16 + 16);
    if (d[3] < 1.5) continue;
    const columns = [
      [d[4], d[5], d[6]],
      [d[8], d[9], d[10]],
      [d[12], d[13], d[14]],
    ];
    const mul = (v) =>
      [0, 1, 2].map((r) => columns.reduce((sum, c, j) => sum + c[r] * v[j], 0));
    const origin = eye.map((x, j) => x - d[j]);
    const a = dot(ray, mul(ray)),
      b = dot(origin, mul(ray)),
      closest = origin.map((v, j) => v - (ray[j] * b) / a),
      delta = 1 - dot(closest, mul(closest));
    if (delta < 0) continue;
    const root = Math.sqrt(delta / a),
      t = -b / a - root;
    if (t < 0.1 || Math.abs(t - nearT) > 0.08) continue;
    const axis = columns.reduce((a, b) => (dot(a, a) > dot(b, b) ? a : b));
    if (Math.abs(dot(norm(axis), normal)) < 0.85) continue;
    expectedThickness += root * 2;
  }
  assert.ok(Math.abs(totalThickness / expectedThickness - 1) < 0.015, {
    totalThickness,
    expectedThickness,
  });
  console.log({ totalThickness, nearestChord, expectedThickness });
  console.log({
    particles: sim.count,
    droplets,
    sheets,
    dropPixelAreas: measured,
    hole,
  });
}
let changed = 0;
for (let i = 0; i < images[0].length; i += 4)
  if (
    Math.abs(images[0][i] - images[1][i]) +
      Math.abs(images[0][i + 1] - images[1][i + 1]) +
      Math.abs(images[0][i + 2] - images[1][i + 2]) >
    8
  )
    changed++;
assert.ok(changed > 500);
await png('/private/tmp/water-details-before.png', images[0]);
await png('/private/tmp/water-details-after.png', images[1]);
// Turning off the experiment clears indirect counts; empty state cannot resurrect old features.
e = device.createCommandEncoder();
volume.encode(e, sim, false);
device.queue.submit([e.finish()]);
assert.equal(new Uint32Array(await read(volume.detailDraw))[1], 0);
sim.count = 0;
e = device.createCommandEncoder();
volume.encode(e, sim, true);
device.queue.submit([e.finish()]);
assert.equal(new Uint32Array(await read(volume.detailDraw))[1], 0);
assert.deepEqual(errors, []);
assert.equal(lost, undefined);
console.log(
  'PASS analytic volumes, holes, coverage, physics untouched, A/B and empty reset',
);
await device.queue.onSubmittedWorkDone();
process.exit(0);
