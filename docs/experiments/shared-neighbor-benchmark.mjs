// Isolate gather execution from task construction and compare identical buffers.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import {
  WebGPUSimulation,
  RENDER_PARAMETER_SLOT,
} from '../../app/webgpu/simulation.ts';
import { sharedNeighbors } from './shared-neighbors.mjs';
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
const p = sim.writeParameters(RENDER_PARAMETER_SLOT, input);
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
for (const width of [32, 64, 128])
  for (const shared of [true, false]) {
    sim.run = originalRun;
    sim.buildGrid = originalGrid;
    const control = await sharedNeighbors(sim, width, shared);
    encoder = device.createCommandEncoder();
    sim.buildGrid(encoder, p);
    device.queue.submit([encoder.finish()]);
    const taskCount = new Uint32Array(await read(control.indirect))[0];
    for (const name of ['lambda', 'prepareVelocity']) {
      const r = name === 'lambda' ? {} : { 5: sim.factor };
      const target = name === 'lambda' ? sim.lambda : sim.factor;
      const outputs = [];
      for (const enabled of [false, true]) {
        control.enabled = enabled;
        control.resetFresh();
        encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        sim.run(pass, name, p, r);
        pass.end();
        device.queue.submit([encoder.finish()]);
        outputs.push(new Float32Array(await read(target)));
      }
      let maxAbs = 0;
      for (let i = 0; i < 50000 * 4; i++)
        maxAbs = Math.max(maxAbs, Math.abs(outputs[0][i] - outputs[1][i]));
      assert.ok(maxAbs < 2e-4, `numeric ${width} ${shared} ${name}: ${maxAbs}`);
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
          for (let k = 0; k < 40; k++) {
            control.resetFresh();
            sim.run(pass, name, p, r);
          }
          pass.end();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          times.push({
            width,
            shared,
            name,
            enabled,
            round,
            queryIndex,
            taskCount,
            maxAbs,
          });
          queryIndex += 2;
        }
    }
    console.log('completed', width, shared, taskCount);
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
  (r) => `${r.width}/${r.shared}/${r.name}/${r.enabled}`,
);
console.log(
  JSON.stringify(
    Array.from(groups, ([key, rows]) => ({
      key,
      ms: rows.reduce((s, r) => s + r.ms, 0) / rows.length,
      maxAbs: rows[0].maxAbs,
      taskCount: rows[0].taskCount,
    })),
    null,
    2,
  ),
);
assert.deepEqual(errors, []);
process.exit(0);
