import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../app/webgpu/simulation.ts';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const device = await (await gpu.requestAdapter()).requestDevice();
const sim = await WebGPUSimulation.create(device),
  errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const kept = (globalThis.readbacks = []);
async function read(b) {
  const out = device.createBuffer({
    size: b.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(b, 0, out, 0, b.size);
  device.queue.submit([e.finish()]);
  await out.mapAsync(GPUMapMode.READ);
  const range = out.getMappedRange();
  kept.push([out, range]);
  const data = new Float32Array(range.slice(0));
  out.unmap();
  return data;
}
function seed(points) {
  let e = device.createCommandEncoder();
  sim.reset(e, 50000);
  device.queue.submit([e.finish()]);
  sim.count = points.length;
  const data = new Float32Array(points.length * 12);
  points.forEach((p, i) => data.set([...p, 1, ...p, 1, 0, 0, 0, i], i * 12));
  device.queue.writeBuffer(sim.state, 0, data);
  device.queue.writeBuffer(
    sim.duck,
    0,
    new Float32Array([0, 10, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
  );
}
const forces = { gravity: 0, viscosity: 0, agitation: 0 };
async function evolve(points, steps = 1, on = true, mode = 'explicit') {
  seed(points);
  for (let i = 0; i < steps; i++) {
    const e = device.createCommandEncoder();
    sim.step(e, { forces, surfaceTension: on, capillaryMode: mode });
    device.queue.submit([e.finish()]);
  }
  const state = (await read(sim.state)).slice(0, points.length * 12);
  assert.ok(state.every(Number.isFinite));
  assert.equal(sim.count, points.length);
  assert.equal(
    new Set(Array.from({ length: sim.count }, (_, i) => state[i * 12 + 11]))
      .size,
    sim.count,
  );
  return state;
}
const points = [
  [-0.04, 0.5, 0],
  [0.04, 0.5, 0],
];
const off = await evolve(points, 1, false),
  on = await evolve(points),
  implicit = await evolve(points, 1, true, 'implicit');
function byId(a, id) {
  for (let i = 0; i < a.length; i += 12)
    if (a[i + 11] === id) return a.slice(i, i + 12);
  throw Error('missing particle');
}
assert.equal(byId(off, 0)[8], 0);
assert.ok(
  byId(on, 0)[8] > 0,
  'cohesion attracts sparse primary particles, not just sheets',
);
assert.ok(byId(implicit, 0)[8] > 0, 'optional implicit path remains valid');
assert.ok(
  Math.abs(byId(on, 0)[8] + byId(on, 1)[8]) < 1e-7,
  'equal/opposite pair impulses',
);
const close = await evolve([
  [-0.008, 0.5, 0],
  [0.008, 0.5, 0],
]);
assert.ok(byId(close, 0)[8] < 0, 'short-range core prevents particle collapse');
const merged = await evolve(points, 120);
const a = byId(merged, 0),
  b = byId(merged, 1),
  distance = Math.hypot(...[0, 1, 2].map((k) => a[k] - b[k]));
console.log({ distance });
assert.ok(
  distance > 0.01 && distance < 0.055,
  'two primary particles approach without spawning or collapsing',
);
const lone = await evolve([[0, 0.5, 0]], 30);
assert.deepEqual(Array.from(lone.slice(8, 11)), [0, 0, 0]);
const patch = [];
for (let x = -3; x <= 3; x++)
  for (let z = -3; z <= 3; z++) patch.push([x * 0.045, 0.5, z * 0.045]);
const surface = await evolve(patch);
const momentum = [0, 0, 0],
  torque = [0, 0, 0];
for (let i = 0; i < surface.length; i += 12) {
  const p = surface.slice(i, i + 3),
    v = surface.slice(i + 8, i + 11);
  for (let k = 0; k < 3; k++) momentum[k] += v[k];
  torque[0] += p[1] * v[2] - p[2] * v[1];
  torque[1] += p[2] * v[0] - p[0] * v[2];
  torque[2] += p[0] * v[1] - p[1] * v[0];
}
assert.ok(Math.hypot(...momentum) < 1e-5);
assert.ok(Math.hypot(...torque) < 1e-5);
assert.deepEqual(errors, []);
console.log({
  initialDistance: 0.08,
  cohesionDistance: distance,
  momentum,
  torque,
});
console.log(
  'PASS general cohesion, repulsion, off/implicit, pair conservation, unchanged primary counts',
);
process.exit(0);
