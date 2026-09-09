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
// Unified field + sub-voxel primary droplet silhouettes. No sheet proxy layer.
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
  assert.equal(
    counts[1],
    drops.length,
    'only existing isolated primaries need analytic silhouettes',
  );
  const features = new Float32Array(await read(volume.details)),
    state = new Float32Array(await read(sim.state)),
    shapes = new Float32Array(await read(volume.shapes));
  const oldRadius = 0.17 * Math.cbrt(0.2) * Math.cbrt(0.1 / 3.6);
  for (let i = 0; i < counts[1]; i++) {
    const d = features.subarray(i * 16, i * 16 + 16);
    assert.equal(d[3], 1, 'no flat sheet proxies');
    assert.ok(
      Math.abs(d[7] / oldRadius - 0.72) < 1e-5,
      'only rendered droplet radius shrinks',
    );
    const source = d[11];
    for (let a = 0; a < 3; a++) assert.equal(d[a], state[source * 12 + a]);
  }
  assert.equal(sim.count, positions.length);
  for (let i = 0; i < sim.count; i++) {
    const id = state[i * 12 + 11];
    for (let a = 0; a < 3; a++)
      assert.equal(state[i * 12 + a], data[id * 12 + a]);
    assert.ok(
      shapes[i * 16 + 15] > 0,
      'every primary contributes to the common density field',
    );
  }
  assert.equal(
    volume.spray,
    undefined,
    'no extra particle simulation is constructed',
  );
  assert.equal(
    renderer.sheetNormal,
    undefined,
    'no sheet-only smoothing target',
  );
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
  const measured = [];
  for (const p of drops) {
    const [cx, cy] = pixel(p);
    let area = 0;
    for (let y = cy - 12; y <= cy + 12; y++)
      for (let x = cx - 12; x <= cx + 12; x++) {
        const at = (x + w * y) * 4;
        if (hits[at + 3] === 1) area += hits[at + 2];
      }
    assert.ok(area > 40 && area < 200);
    measured.push(area);
  }
  console.log({
    primaryParticles: sim.count,
    analyticPrimaries: counts[1],
    radiusRatio: 0.72,
    renderedVolumeRatio: 0.72 ** 3,
    pixelAreas: measured,
  });
}
await png('/private/tmp/water-unified-before.png', images[0]);
await png('/private/tmp/water-unified-after.png', images[1]);
const center = [
  -2.08 + (64 / 127) * 4.16,
  -1.12 + (50 / 159) * 5.2,
  -1.56 + (48 / 95) * 3.12,
];
async function pair(distance) {
  sim.count = 2;
  const a = new Float32Array(24);
  for (let i = 0; i < 2; i++) {
    const p = [center[0] + (i - 0.5) * distance, center[1], center[2]];
    a.set([...p, 1, ...p, 1, 0, 0, 0, i], i * 12);
  }
  device.queue.writeBuffer(sim.state, 0, a);
  e = device.createCommandEncoder();
  volume.encode(e, sim, true);
  device.queue.submit([e.finish()]);
  const density = new Float32Array(await read(volume.density));
  const count = new Uint32Array(await read(volume.detailDraw))[1];
  assert.ok(count <= 2);
  return density[64 + 128 * (50 + 160 * 48)];
}
const separated = await pair(0.14),
  merged = await pair(0.045);
assert.equal(
  separated,
  0,
  'separated primary drops have no artificial connecting layer',
);
assert.ok(
  merged > 1.15,
  'nearby primary kernels form a common surface through their midpoint',
);
console.log({
  separatedMidpointDensity: separated,
  mergedMidpointDensity: merged,
  iso: 1.15,
});
sim.count = 0;
e = device.createCommandEncoder();
volume.encode(e, sim, true);
device.queue.submit([e.finish()]);
assert.equal(new Uint32Array(await read(volume.detailDraw))[1], 0);
assert.ok(new Float32Array(await read(volume.density)).every((v) => v === 0));
assert.deepEqual(errors, []);
assert.equal(lost, undefined);
console.log(
  'PASS unchanged main particles, smaller visible drops, unified density fusion, no sheet/spray layer, empty reset',
);
process.exit(0);
