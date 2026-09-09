import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../app/webgpu/volume.ts';
Object.assign(globalThis, globals);
const gpu = create(['backend=metal']);
globalThis.nativeGPU = gpu;
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
let lost;
device.lost.then((info) => {
  lost = info;
});
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const retainedReadbacks = (globalThis.retainedReadbacks = []);
async function read(buffer) {
  const b = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(buffer, 0, b, 0, buffer.size);
  device.queue.submit([e.finish()]);
  await b.mapAsync(GPUMapMode.READ);
  const range = b.getMappedRange();
  const a = new Float32Array(range.slice(0));
  b.unmap();
  retainedReadbacks.push([b, range]);
  return a;
}
const sim = await WebGPUSimulation.create(device);
const volume = await WebGPUVolume.create(device);
let e = device.createCommandEncoder();
sim.reset(e, 50000);
device.queue.submit([e.finish()]);
let initial = await read(sim.state);
assert.equal(
  new Set(Array.from({ length: 50000 }, (_, i) => initial[i * 12 + 11])).size,
  50000,
);
for (let frame = 0; frame < 240; frame++) {
  e = device.createCommandEncoder();
  sim.step(e, { forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 } });
  device.queue.submit([e.finish()]);
  if (frame % 60 === 59) {
    await device.queue.onSubmittedWorkDone();
    console.log(
      'physics step',
      frame + 1,
      'duck',
      Array.from((await read(sim.duck)).slice(0, 3)),
    );
  }
}
const state = await read(sim.state);
assert.ok(state.every(Number.isFinite));
assert.equal(
  new Set(Array.from({ length: 50000 }, (_, i) => state[i * 12 + 11])).size,
  50000,
);
let energy = 0;
for (let i = 0; i < 50000; i++) {
  const p = state.subarray(i * 12, i * 12 + 3);
  assert.ok(
    p[0] >= -1.78001 &&
      p[0] <= 1.78001 &&
      p[1] >= -0.91701 &&
      p[1] <= 3.80001 &&
      p[2] >= -1.28001 &&
      p[2] <= 1.28001,
  );
  energy +=
    state[i * 12 + 8] ** 2 + state[i * 12 + 9] ** 2 + state[i * 12 + 10] ** 2;
}
console.log('mean speed squared', energy / 50000);
assert.ok(energy / 50000 < 1);
// Exercise a strong impulse, wall motion and stirring before rebuilding the field.
for (let frame = 0; frame < 120; frame++) {
  e = device.createCommandEncoder();
  sim.step(e, {
    forces: { gravity: 9.8, viscosity: 0.025, agitation: 0.5 },
    splash: frame === 0 ? [0.6, 0.3, 1.5] : undefined,
    shake: frame < 30 ? 8 : 0,
    brush:
      frame < 30
        ? { x: 0, y: -0.5, z: 0, dx: 2, dz: 1, strength: 1, mode: 'stir' }
        : undefined,
  });
  device.queue.submit([e.finish()]);
  if (frame % 2 === 1) await device.queue.onSubmittedWorkDone();
}
const moving = await read(sim.state);
assert.ok(moving.every(Number.isFinite));
const duck = await read(sim.duck);
assert.ok(duck.every(Number.isFinite));
assert.ok(Math.abs(Math.hypot(...duck.slice(4, 8)) - 1) < 0.001);
e = device.createCommandEncoder();
volume.encode(e, sim, false);
console.log('density encoded');
device.queue.submit([e.finish()]);
await device.queue.onSubmittedWorkDone();
console.log('density finished');
console.log('reading density');
const density = await read(volume.density);
console.log('density read', density.length);
assert.ok(density.every(Number.isFinite));
assert.ok(density.some((x) => x > 1.15));
console.log('reading shapes');
const shapes = await read(volume.shapes);
console.log('shapes read');
const scale = Math.cbrt(10000 / 50000);
for (const [x, y, z] of [
  [64, 11, 48],
  [64, 15, 48],
  [33, 11, 40],
  [64, 80, 48],
  [12, 9, 12],
  [115, 10, 84],
  [32, 20, 30],
  [78, 14, 53],
  [64, 23, 48],
  [92, 17, 64],
  [18, 19, 34],
  [103, 26, 79],
]) {
  const p = [
    -2.08 + (x / 127) * 4.16,
    -1.12 + (y / 159) * 5.2,
    -1.56 + (z / 95) * 3.12,
  ];
  let expected = 0;
  for (let i = 0; i < 50000; i++) {
    const s = shapes.subarray(i * 16, i * 16 + 16);
    let b = Math.max(0, Math.min(1, (s[3] - 0.15) / 1.05));
    b = b * b * (3 - 2 * b);
    const r = (0.1 + 0.09 * b) * scale;
    const d = p.map((v, a) => (v - s[a]) / r);
    let r2 = 0;
    for (let row = 0; row < 3; row++)
      for (let col = 0; col < 3; col++)
        r2 += d[row] * s[4 + col * 4 + row] * d[col];
    if (r2 < 1) expected += (1 - r2) ** 3 * (1 + Math.max(0, 1 - s[3]) * 0.8);
  }
  const actual = density[x + 128 * (y + 160 * z)];
  assert.ok(
    Math.abs(actual - expected) < Math.max(0.0002, expected * 0.0002),
    JSON.stringify({ x, y, z, actual, expected }),
  );
  console.log('density oracle', actual, expected);
}
// Every final-position particle must be in its exact cell range after scatter.
const ordered = await read(sim.state);
const startsFloat = await read(sim.starts);
const starts = new Uint32Array(startsFloat.buffer);
assert.equal(starts[30720], 50000);
for (let k = 0; k < 30720; k++) {
  assert.ok(starts[k] <= starts[k + 1]);
  for (let i = starts[k]; i < starts[k + 1]; i++) {
    const p = ordered.subarray(i * 12, i * 12 + 3);
    const c = p.map((v, a) =>
      Math.max(
        0,
        Math.min(
          [31, 39, 23][a],
          Math.floor((v - [-2.04, -1.19, -1.53][a]) / (0.225 * scale)),
        ),
      ),
    );
    assert.equal(c[0] + 32 * (c[1] + 40 * c[2]), k);
  }
}
// Empty grid/reduction and re-injection after draining must never retain stale water.
sim.count = 0;
e = device.createCommandEncoder();
sim.step(e, { forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 } });
volume.encode(e, sim, false);
device.queue.submit([e.finish()]);
assert.ok((await read(volume.density)).every((v) => v === 0));
sim.count = 18;
e = device.createCommandEncoder();
sim.step(e, {
  forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 },
  previousCount: 0,
  pourAt: [0.4, 0.3],
});
device.queue.submit([e.finish()]);
const poured = await read(sim.state);
for (let i = 0; i < 18; i++) assert.ok(poured[i * 12 + 1] > 1.7);
for (const quality of [15000, 30000, 50000]) {
  e = device.createCommandEncoder();
  sim.reset(e, quality);
  sim.step(e, { forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 } }, 0);
  sim.step(e, { forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 } }, 1);
  volume.encode(e, sim, false);
  device.queue.submit([e.finish()]);
  const a = await read(sim.state);
  assert.ok(a.slice(0, quality * 12).every(Number.isFinite));
  assert.equal(
    new Set(Array.from({ length: quality }, (_, i) => a[i * 12 + 11])).size,
    quality,
  );
}
assert.equal(lost, undefined, JSON.stringify(lost));
assert.deepEqual(errors, []);
console.log(
  'PASS: 50k stable/disturbed motion, exact grid, density oracle, empty/reinjection, all quality levels, two substeps',
);
// Retain native Dawn mapped-buffer wrappers through process teardown.
await device.queue.onSubmittedWorkDone();
process.exit(0);
