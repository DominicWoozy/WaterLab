// Isolate gather execution from task construction and compare identical buffers.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import {
  WebGPUSimulation,
  RENDER_PARAMETER_SLOT,
} from '../../app/webgpu/simulation.ts';
import { specializedKernels } from './specialized-kernels.mjs';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const device = await (
  await gpu.requestAdapter()
).requestDevice({ requiredFeatures: ['timestamp-query'] });
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const sim = await WebGPUSimulation.create(device);
const input = { forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 } };
let encoder = device.createCommandEncoder();
sim.reset(encoder, 50000);
device.queue.submit([encoder.finish()]);
for (let i = 0; i < 120; i++) {
  encoder = device.createCommandEncoder();
  sim.step(encoder, input);
  device.queue.submit([encoder.finish()]);
  if (i % 2) await device.queue.onSubmittedWorkDone();
}
const originalRun = sim.run.bind(sim),
  originalGrid = sim.buildGrid.bind(sim);

const times = [];
const query = device.createQuerySet({ type: 'timestamp', count: 200 });
let queryIndex = 0;
const kept = (globalThis.readbacks = []);
async function read(b) {
  const staging = device.createBuffer({
    size: b.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(b, 0, staging, 0, b.size);
  device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const range = staging.getMappedRange();
  kept.push([staging, range]);
  const data = range.slice(0);
  staging.unmap();
  return data;
}
for (const prune of [false, true]) {
  sim.run = originalRun;
  sim.buildGrid = originalGrid;
  const control = await specializedKernels(sim, { prune });
  for (const mode of [0, 1, 2]) {
    const parameter = sim.writeParameters(RENDER_PARAMETER_SLOT, {
      ...input,
      surfaceTension: mode !== 0,
      capillaryMode: mode === 2 ? 'implicit' : 'explicit',
    });
    for (let round = 0; round < 4; round++)
      for (const enabled of round % 2 ? [true, false] : [false, true]) {
        control.enabled = enabled;
        encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass({
          timestampWrites: {
            querySet: query,
            beginningOfPassWriteIndex: queryIndex,
            endOfPassWriteIndex: queryIndex + 1,
          },
        });
        for (let i = 0; i < 40; i++) sim.run(pass, 'viscosity', parameter);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        times.push({ prune, mode, enabled, round, queryIndex });
        queryIndex += 2;
      }
  }
}
const resolved = device.createBuffer({
  size: queryIndex * 8,
  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
});
encoder = device.createCommandEncoder();
encoder.resolveQuerySet(query, 0, queryIndex, resolved, 0);
device.queue.submit([encoder.finish()]);
const values = new BigUint64Array(await read(resolved));
for (const row of times)
  row.ms =
    Number(values[row.queryIndex + 1] - values[row.queryIndex]) / 1e6 / 40;
const groups = Map.groupBy(
  times.filter((r) => r.round > 0),
  (r) => `${r.prune}/${r.mode}/${r.enabled}`,
);
console.log(
  JSON.stringify(
    Array.from(groups, ([key, rows]) => ({
      key,
      ms: rows.reduce((s, r) => s + r.ms, 0) / rows.length,
    })),
    null,
    2,
  ),
);
assert.deepEqual(errors, []);
process.exit(0);
