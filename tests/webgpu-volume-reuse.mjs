// Compare optimized reconstruction to the full rebuild using identical states.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import {
  WebGPUSimulation,
  RENDER_PARAMETER_SLOT,
} from '../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../app/webgpu/volume.ts';
import {
  NEIGHBOR_CACHE_BASE,
  NEIGHBOR_CACHE_SIZE,
} from '../app/webgpu/common.ts';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const device = await (await gpu.requestAdapter()).requestDevice();
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const sim = await WebGPUSimulation.create(device),
  volume = await WebGPUVolume.create(device);
const kept = (globalThis.readbacks = []);
async function read(b, bytes = b.size) {
  const staging = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(b, 0, staging, 0, bytes);
  device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const range = staging.getMappedRange();
  kept.push([staging, range]);
  const result = new Float32Array(range.slice(0));
  staging.unmap();
  return result;
}
function compare(a, b, label) {
  assert.equal(a.length, b.length);
  let max = 0,
    sum = 0;
  for (let i = 0; i < a.length; i++) {
    const error = Math.abs(a[i] - b[i]);
    assert.ok(Number.isFinite(error), label + ' finite');
    max = Math.max(max, error);
    sum += error * error;
    assert.ok(
      error <
        Math.max(
          label.endsWith('filtered') ? 3e-4 : 3e-5,
          Math.abs(b[i]) * 3e-6,
        ),
      `${label}[${i}]: ${a[i]} != ${b[i]}`,
    );
  }
  return { max, rms: Math.sqrt(sum / a.length) };
}
async function snapshot() {
  return {
    raw: await read(volume.density),
    filtered: await read(volume.temp),
    shapes: await read(volume.shapes, sim.count * 64),
    details: await read(volume.details),
    draw: new Uint32Array((await read(volume.detailDraw)).buffer),
    state: await read(sim.state, sim.count * 48),
  };
}
async function verify(label) {
  const original = await read(sim.state, sim.count * 48);
  let e = device.createCommandEncoder();
  volume.encode(e, sim, true, true);
  device.queue.submit([e.finish()]);
  const fast = await snapshot();
  assert.deepEqual(
    fast.state,
    original,
    'reuse does not reorder or change particle state',
  );
  e = device.createCommandEncoder();
  volume.encode(e, sim, true);
  device.queue.submit([e.finish()]);
  const full = await snapshot();
  const raw = compare(fast.raw, full.raw, label + ' raw'),
    filtered = compare(fast.filtered, full.filtered, label + ' filtered');
  const ids = new Map();
  for (let i = 0; i < sim.count; i++) ids.set(full.state[i * 12 + 11], i);
  for (let i = 0; i < sim.count; i++) {
    const j = ids.get(fast.state[i * 12 + 11]);
    assert.notEqual(j, undefined);
    assert.deepEqual(
      fast.state.slice(i * 12, i * 12 + 12),
      full.state.slice(j * 12, j * 12 + 12),
      'all physical particle values preserved',
    );
    compare(
      fast.shapes.slice(i * 16, i * 16 + 16),
      full.shapes.slice(j * 16, j * 16 + 16),
      'particle shape',
    );
  }
  assert.equal(
    fast.draw[1],
    full.draw[1],
    'same isolated primary droplet count',
  );
  const detailsById = new Map();
  for (let i = 0; i < full.draw[1]; i++) {
    const d = full.details.slice(i * 16, i * 16 + 16);
    const id = full.state[d[11] * 12 + 11];
    d[11] = id;
    detailsById.set(id, d);
  }
  for (let i = 0; i < fast.draw[1]; i++) {
    const d = fast.details.slice(i * 16, i * 16 + 16);
    const id = fast.state[d[11] * 12 + 11];
    d[11] = id;
    compare(d, detailsById.get(id), 'analytic droplet');
  }
  console.log({ label, raw, filtered, drops: fast.draw[1] });
}
const input = {
  forces: { gravity: 9.8, viscosity: 0.025, agitation: 1.4 },
  splash: [0.5, 0.3, 2],
};
for (const quality of [15000, 30000, 50000, 70000, 100000]) {
  let e = device.createCommandEncoder();
  sim.reset(e, quality);
  device.queue.submit([e.finish()]);
  for (let frame = 0; frame < 5; frame++) {
    e = device.createCommandEncoder();
    sim.step(e, { ...input, splash: frame === 0 ? input.splash : undefined });
    device.queue.submit([e.finish()]);
  }
  await verify(String(quality));
}
// Dense cache overflow and small isolated drops exercise both geometry paths.
let e = device.createCommandEncoder();
sim.reset(e, 50000);
device.queue.submit([e.finish()]);
sim.count = 301;
const data = new Float32Array(sim.count * 12);
for (let i = 0; i < sim.count; i++) {
  const p =
    i === 300
      ? [1.2, 0.5, 0.8]
      : [
          (i % 10) * 0.004,
          0.5 + (Math.floor(i / 10) % 10) * 0.004,
          Math.floor(i / 100) * 0.004,
        ];
  data.set([...p, 1, ...p, 1, 0, 0, 0, i], i * 12);
}
device.queue.writeBuffer(sim.state, 0, data);
const params = sim.writeParameters(RENDER_PARAMETER_SLOT, input);
e = device.createCommandEncoder();
sim.buildGrid(e, params);
const pass = e.beginComputePass();
sim.run(pass, 'prepareVelocity', params, { 5: sim.factor });
pass.end();
sim.swap();
device.queue.submit([e.finish()]);
const starts = new Uint32Array((await read(sim.starts)).buffer);
assert.ok(
  starts
    .slice(NEIGHBOR_CACHE_BASE, NEIGHBOR_CACHE_BASE + sim.count)
    .some((n) => n > NEIGHBOR_CACHE_SIZE),
  'overflow exercised',
);
await verify('overflow and isolated drop');
// A drain-only frame must use the default fresh-grid path and clear the field.
sim.count = 0;
e = device.createCommandEncoder();
volume.encode(e, sim, true);
device.queue.submit([e.finish()]);
assert.ok(
  (await read(volume.temp)).every((x) => x === 0),
  'empty reconstruction clears old water',
);
assert.deepEqual(errors, []);
console.log(
  'PASS exact-state reconstruction reuse, 15k/30k/50k/70k/100k, full fields, shapes, primary drops, overflow and drain',
);
process.exit(0);
