import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import {
  WebGPUSimulation,
  RENDER_PARAMETER_SLOT,
} from '../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../app/webgpu/volume.ts';
import { CAPACITY, NEIGHBOR_CACHE_BASE } from '../app/webgpu/common.ts';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const device = await (await gpu.requestAdapter()).requestDevice();
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const sim = await WebGPUSimulation.create(device),
  volume = await WebGPUVolume.create(device);
const kept = (globalThis.readbacks = []);
async function read(b) {
  const staging = device.createBuffer({
    size: b.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(b, 0, staging, 0, b.size);
  device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const range = staging.getMappedRange();
  kept.push([staging, range]);
  const result = new Float32Array(range.slice(0));
  staging.unmap();
  return result;
}
const input = { forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 } };
for (const quality of [70000, 100000, 50000, 100000]) {
  let e = device.createCommandEncoder();
  sim.reset(e, quality);
  device.queue.submit([e.finish()]);
  let state = await read(sim.state);
  assert.equal(sim.count, quality);
  assert.equal(
    new Set(Array.from({ length: quality }, (_, i) => state[i * 12 + 11])).size,
    quality,
  );
  for (let i = 0; i < quality; i++) {
    const p = state.subarray(i * 12, i * 12 + 3);
    assert.ok(
      p[0] >= -1.701 &&
        p[0] <= 1.701 &&
        p[1] >= -0.918 &&
        p[1] <= -0.556 &&
        Math.abs(p[2]) <= 1.201,
    );
  }
  // A synthetic exact sum isolates the >256 group reduction from buoyancy.
  const reactions = new Float32Array(quality * 8);
  for (let i = 0; i < quality; i++)
    reactions.set([1, 2, 0, 0, 0, 0, 3, 0], i * 8);
  device.queue.writeBuffer(sim.reactions, 0, reactions);
  const p = sim.writeParameters(RENDER_PARAMETER_SLOT, input);
  e = device.createCommandEncoder();
  const pass = e.beginComputePass();
  sim.run(
    pass,
    'reduceParticles',
    p,
    { 1: sim.reactions, 2: sim.reactionGroups },
    Math.ceil(quality / 256),
  );
  sim.run(
    pass,
    'reduceGroups',
    p,
    { 1: sim.reactionGroups, 2: sim.reactionTotal },
    1,
  );
  pass.end();
  device.queue.submit([e.finish()]);
  const total = await read(sim.reactionTotal);
  assert.deepEqual(Array.from(total.slice(0, 8)), [
    quality,
    quality * 2,
    0,
    0,
    0,
    0,
    quality * 3,
    0,
  ]);
  // Reset synthetic impulses before a real step, then test exact counting/sorting.
  e = device.createCommandEncoder();
  sim.reset(e, quality);
  sim.step(e, input);
  device.queue.submit([e.finish()]);
  state = await read(sim.state);
  assert.ok(state.subarray(0, quality * 12).every(Number.isFinite));
  const ids = Array.from({ length: quality }, (_, i) => state[i * 12 + 11]);
  assert.equal(new Set(ids).size, quality);
  assert.equal(Math.max(...ids), quality - 1);
  const grid = new Uint32Array((await read(sim.starts)).buffer);
  assert.equal(grid[NEIGHBOR_CACHE_BASE - 1], quality);
  for (let c = 0; c < NEIGHBOR_CACHE_BASE - 1; c++)
    assert.ok(grid[c] <= grid[c + 1]);
  console.log('initialization, reduction and full step', quality);
}

for (const quality of [70000, 100000]) {
  let e = device.createCommandEncoder();
  sim.reset(e, quality);
  device.queue.submit([e.finish()]);
  // Exercise the region beyond the old high-quality grid's upper X/Y extent.
  const points = [];
  for (const centre of [
    [1.73, 3.74, 1.23],
    [-1.73, -0.87, -1.23],
    [0, 1, 0],
  ])
    for (let i = 0; i < 8; i++)
      points.push(centre.map((v, k) => v + ((i >> k) & 1) * 0.015));
  sim.count = points.length;
  const data = new Float32Array(sim.count * 12);
  points.forEach((p, i) => data.set([...p, 1, ...p, 1, 0, 0, 0, i], i * 12));
  device.queue.writeBuffer(sim.state, 0, data);
  device.queue.writeBuffer(
    sim.duck,
    0,
    new Float32Array([0, 10, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
  );
  const p = sim.writeParameters(RENDER_PARAMETER_SLOT, input);
  e = device.createCommandEncoder();
  sim.buildGrid(e, p);
  let pass = e.beginComputePass();
  sim.run(pass, 'factor', p, { 4: sim.lambda, 5: sim.factor });
  pass.end();
  device.queue.submit([e.finish()]);
  const reference = await read(sim.factor),
    sorted = await read(sim.state);
  e = device.createCommandEncoder();
  pass = e.beginComputePass();
  sim.run(pass, 'prepareVelocity', p, { 5: sim.factor });
  pass.end();
  device.queue.submit([e.finish()]);
  const actual = await read(sim.factor),
    grid = new Uint32Array((await read(sim.starts)).buffer);
  const h = 0.17 * Math.cbrt(10000 / quality);
  for (let i = 0; i < sim.count; i++) {
    let count = 0;
    for (let j = 0; j < sim.count; j++) {
      const r = Math.hypot(
        ...[0, 1, 2].map((k) => sorted[i * 12 + k] - sorted[j * 12 + k]),
      );
      if (r < h && r > 1e-6) count++;
    }
    assert.equal(
      grid[NEIGHBOR_CACHE_BASE + i],
      count,
      'boundary neighbors cannot be culled',
    );
    for (let k = 0; k < 4; k++)
      assert.ok(Math.abs(reference[i * 4 + k] - actual[i * 4 + k]) < 2e-4);
  }
  // Independently gather density at a voxel close to the upper-right cluster.
  e = device.createCommandEncoder();
  volume.encode(e, sim, true, false);
  device.queue.submit([e.finish()]);
  const shapes = await read(volume.shapes),
    field = await read(volume.density);
  const xyz = [1.75, 3.75, 1.25],
    lo = [-2.08, -1.12, -1.56],
    hi = [2.08, 4.08, 1.56],
    size = [128, 160, 96];
  const voxel = xyz.map((v, k) =>
    Math.round(((v - lo[k]) / (hi[k] - lo[k])) * (size[k] - 1)),
  );
  const world = voxel.map(
    (v, k) => lo[k] + (v / (size[k] - 1)) * (hi[k] - lo[k]),
  );
  let expected = 0;
  for (let i = 0; i < sim.count; i++) {
    const o = i * 16,
      d = world.map((v, k) => (v - shapes[o + k]) / shapes[o + 3]);
    const md = [0, 1, 2].map(
      (k) =>
        d[0] * shapes[o + 4 + k] +
        d[1] * shapes[o + 8 + k] +
        d[2] * shapes[o + 12 + k],
    );
    const r2 = d.reduce((s, v, k) => s + v * md[k], 0);
    if (r2 < 1) expected += (1 - r2) ** 3 * shapes[o + 7];
  }
  const index = voxel[0] + 128 * (voxel[1] + 160 * voxel[2]);
  assert.ok(expected > 0.01, 'test point must intersect water');
  // Low-density sparse features bypass surface filtering.
  assert.ok(
    Math.abs(field[index] - expected) < 0.002,
    `edge density ${field[index]} vs ${expected}`,
  );
  console.log('container boundary neighbors and density', quality);
}
assert.equal(CAPACITY, 100000);
assert.deepEqual(errors, []);
console.log(
  'PASS 70k/100k capacity, initialization, mode changes, u32 IDs, full buoyancy reduction and domain coverage',
);
process.exit(0);
