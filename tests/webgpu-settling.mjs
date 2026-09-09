// Native GPU regression: bounded energy alone does not detect a jittering tank.
// Match particles by ID because every grid rebuild may permute their order.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { WebGPUSimulation, PHYSICS_DT } from '../app/webgpu/simulation.ts';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const device = await (await gpu.requestAdapter()).requestDevice();
const sim = await WebGPUSimulation.create(device);
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const kept = (globalThis.readbacks = []);
async function read(buffer) {
  const staging = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const range = staging.getMappedRange();
  kept.push([staging, range]);
  const data = new Float32Array(range.slice(0));
  staging.unmap();
  return data;
}
function ordered(data) {
  const result = new Float32Array(sim.count * 6),
    seen = new Set();
  for (let i = 0; i < sim.count; i++) {
    const offset = i * 12,
      id = data[offset + 11];
    assert.ok(Number.isInteger(id) && id >= 0 && id < sim.count);
    seen.add(id);
    const p = data.subarray(offset, offset + 3);
    assert.ok(
      p[0] >= -1.78001 &&
        p[0] <= 1.78001 &&
        p[1] >= -0.91701 &&
        p[1] <= 3.80001 &&
        p[2] >= -1.28001 &&
        p[2] <= 1.28001,
    );
    result.set(p, id * 6);
    result.set(data.subarray(offset + 8, offset + 11), id * 6 + 3);
  }
  assert.equal(seen.size, sim.count);
  assert.ok(result.every(Number.isFinite));
  return result;
}
function measure(now, previous = now) {
  let speed2 = 0,
    movement2 = 0,
    change2 = 0;
  for (let i = 0; i < now.length; i += 6)
    for (let axis = 0; axis < 3; axis++) {
      speed2 += now[i + axis + 3] ** 2;
      movement2 += (now[i + axis] - previous[i + axis]) ** 2;
      change2 += (now[i + axis + 3] - previous[i + axis + 3]) ** 2;
    }
  return {
    rmsSpeed: Math.sqrt(speed2 / sim.count),
    rmsMovement: Math.sqrt(movement2 / sim.count),
    rmsVelocityChange: Math.sqrt(change2 / sim.count),
  };
}
const forces = { gravity: 9.8, viscosity: 0.025, agitation: 0 };
async function advance(frames, input, disturb = false) {
  for (let frame = 0; frame < frames; frame++) {
    const encoder = device.createCommandEncoder();
    sim.step(encoder, {
      ...input,
      ...(disturb
        ? {
            shake: frame < 30 ? 8 : 0,
            splash: frame === 0 ? [0.6, 0.3, 1.5] : undefined,
          }
        : {}),
    });
    device.queue.submit([encoder.finish()]);
    if (frame % 60 === 59) await device.queue.onSubmittedWorkDone();
  }
}
async function window(input) {
  let previous = ordered(await read(sim.state));
  const rows = [];
  for (let frame = 0; frame < 24; frame++) {
    await advance(1, input);
    const now = ordered(await read(sim.state));
    rows.push(measure(now, previous));
    previous = now;
  }
  const mean = Object.fromEntries(
    Object.keys(rows[0]).map((key) => [
      key,
      rows.reduce((sum, row) => sum + row[key], 0) / rows.length,
    ]),
  );
  // These are solver-density errors, not the density used for rendering.
  const factor = await read(sim.factor);
  let compression2 = 0;
  for (let i = 0; i < sim.count; i++)
    compression2 += Math.max(0, factor[i * 4 + 1] - 1) ** 2;
  mean.compressionRms = Math.sqrt(compression2 / sim.count);
  return mean;
}
function settled(stats, label) {
  assert.ok(
    stats.rmsSpeed < 0.02,
    `${label}: residual speed ${stats.rmsSpeed}`,
  );
  assert.ok(
    stats.rmsMovement < 0.0004,
    `${label}: position jitter ${stats.rmsMovement}`,
  );
  assert.ok(
    stats.rmsVelocityChange < 0.006,
    `${label}: velocity jitter ${stats.rmsVelocityChange}`,
  );
  // A softer, highly compressed tank must not pass just because it stops moving.
  // The fidelity preset must keep RMS compression below 1% at equilibrium.
  assert.ok(
    stats.compressionRms < 0.01,
    `${label}: compression ${stats.compressionRms}`,
  );
}
for (const [quality, tension] of [
  [50000, true],
  [50000, false],
  [30000, true],
  [15000, true],
]) {
  const encoder = device.createCommandEncoder();
  sim.reset(encoder, quality);
  device.queue.submit([encoder.finish()]);
  const input = { forces, surfaceTension: tension };
  await advance(300, input);
  const early = measure(ordered(await read(sim.state)));
  await advance(600, input);
  const late = await window(input);
  const label = `${quality} tension=${tension}`;
  console.log(JSON.stringify({ label, seconds: 15, early, late }));
  settled(late, label);
  if (quality === 50000 && tension) {
    let support = 0;
    for (let sample = 0; sample < 30; sample++) {
      await advance(1, input);
      const impulse = await read(sim.reactionTotal);
      support += impulse[1] / PHYSICS_DT;
    }
    support /= 30;
    const duck = await read(sim.duck);
    const expectedWeight = 0.026 * forces.gravity;
    assert.ok(duck[1] > -0.7 && duck[1] < 0, 'duck must float above the floor');
    assert.ok(
      Math.abs(support / expectedWeight - 1) < 0.1,
      'mean buoyant support must balance duck weight',
    );
    console.log(
      JSON.stringify({
        label: 'floating equilibrium',
        support,
        expectedWeight,
        duckY: duck[1],
      }),
    );
  }

  assert.ok(
    late.rmsSpeed < Math.max(0.006, early.rmsSpeed * 0.85),
    `${label}: motion must decay`,
  );
  if (quality === 50000 && tension) {
    await advance(60, input, true);
    const disturbed = measure(ordered(await read(sim.state)));
    assert.ok(
      disturbed.rmsSpeed > Math.max(0.08, late.rmsSpeed * 5),
      'stirring must still create real motion',
    );
    await advance(900, input);
    const fading = await window(input);
    assert.ok(
      fading.rmsSpeed < disturbed.rmsSpeed * 0.15,
      'waves must decay after the impulse',
    );
    // Large coherent waves may outlive a calm reset. Allow 30 seconds after
    // disturbing the tank, retaining the same final jitter limits.
    await advance(900, input);
    const recovered = await window(input);
    console.log(
      JSON.stringify({
        label: '50k after disturbance',
        disturbed,
        fading,
        recovered,
      }),
    );
    settled(recovered, 'after disturbance');
  }
}
assert.deepEqual(errors, []);
console.log(
  'PASS: calm and post-impulse settling, tension on/off, all qualities, bounded density error, particle IDs and bounds',
);
process.exit(0);
