// Independent all-pairs checks for cached neighbors, overflow and fused factors.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import {
  WebGPUSimulation,
  PHYSICS_DT,
  RENDER_PARAMETER_SLOT,
} from '../app/webgpu/simulation.ts';
import {
  CAPACITY,
  NEIGHBOR_CACHE_BASE,
  NEIGHBOR_CACHE_SIZE,
} from '../app/webgpu/common.ts';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const device = await (await gpu.requestAdapter()).requestDevice();
const sim = await WebGPUSimulation.create(device),
  errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const kept = (globalThis.readbacks = []);
async function read(buffer) {
  const staging = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
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
const input = {
  forces: { gravity: 0, viscosity: 0.025, agitation: 0 },
  surfaceTension: false,
};
const h = Math.fround(0.17 * Math.fround(Math.cbrt(0.2)));
for (const [name, spacing] of [
  ['sparse', 0.071],
  ['overflow', 0.006],
]) {
  let encoder = device.createCommandEncoder();
  sim.reset(encoder, 50000);
  device.queue.submit([encoder.finish()]);
  const n = 513;
  sim.count = n;
  const data = new Float32Array(n * 12);
  for (let i = 0; i < n; i++) {
    // Offset the lattice across grid faces; keep it clear of walls and the duck.
    const p =
      i === 512
        ? [1.5, 2, 1]
        : [
            0.065 + ((i % 8) - 3.5) * spacing,
            0.65 + ((Math.floor(i / 8) % 8) - 3.5) * spacing,
            0.049 + (Math.floor(i / 64) - 3.5) * spacing,
          ];
    const v = [Math.sin(i) * 0.05, Math.cos(i) * 0.04, Math.sin(i * 3) * 0.03];
    data.set(
      [...p, 1, ...p.map((x, a) => x - v[a] * PHYSICS_DT), 1, ...v, i],
      i * 12,
    );
  }
  device.queue.writeBuffer(sim.state, 0, data);
  device.queue.writeBuffer(
    sim.duck,
    0,
    new Float32Array([0, 10, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
  );
  const params = sim.writeParameters(RENDER_PARAMETER_SLOT, input);
  encoder = device.createCommandEncoder();
  sim.buildGrid(encoder, params);
  device.queue.submit([encoder.finish()]);
  const sorted = await read(sim.state);
  // Original full-grid kernels remain a GPU reference for the fused pass.
  encoder = device.createCommandEncoder();
  let pass = encoder.beginComputePass();
  sim.run(pass, 'velocity', params);
  sim.run(pass, 'factor', params, { 4: sim.lambda, 5: sim.factor });
  pass.end();
  device.queue.submit([encoder.finish()]);
  const referenceVelocity = await read(sim.spare),
    referenceFactors = await read(sim.factor),
    referenceSurface = await read(sim.surface);
  encoder = device.createCommandEncoder();
  pass = encoder.beginComputePass();
  sim.run(pass, 'prepareVelocity', params, { 5: sim.factor });
  pass.end();
  device.queue.submit([encoder.finish()]);
  const prepared = await read(sim.spare),
    factors = await read(sim.factor),
    surface = await read(sim.surface);
  const cache = new Uint32Array((await read(sim.starts)).buffer);
  let overflowing = 0,
    empty = 0;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 12; k++)
      assert.ok(
        Math.abs(prepared[i * 12 + k] - referenceVelocity[i * 12 + k]) < 1e-6,
        'fused velocity must match original',
      );
    for (let k = 0; k < 4; k++) {
      assert.ok(
        Math.abs(factors[i * 4 + k] - referenceFactors[i * 4 + k]) < 2e-4,
        'fused factor must match full grid',
      );
      assert.ok(
        Math.abs(surface[i * 4 + k] - referenceSurface[i * 4 + k]) < 2e-4,
        'fused surface must match full grid',
      );
    }
    const expected = [];
    let rho = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const r = Math.hypot(
        ...[0, 1, 2].map((k) => sorted[i * 12 + k] - sorted[j * 12 + k]),
      );
      if (r >= h || r < 1e-6) continue;
      expected.push(j);
      rho += (1 - r / h) ** 2;
    }
    const count = cache[NEIGHBOR_CACHE_BASE + i];
    assert.equal(count, expected.length, 'all-pairs neighbor count');
    assert.ok(
      Math.abs(factors[i * 4 + 1] - rho / 3.6) < 2e-4,
      'independent all-pairs density',
    );
    if (count === 0) empty++;
    if (count > NEIGHBOR_CACHE_SIZE) {
      overflowing++;
      continue;
    }
    const actual = Array.from(
      { length: count },
      (_, entry) =>
        cache[NEIGHBOR_CACHE_BASE + CAPACITY + entry * CAPACITY + i],
    );
    assert.deepEqual(
      actual.sort((a, b) => a - b),
      expected,
      'cached neighbor set must be complete, unique and exclude self',
    );
  }
  assert.ok(empty > 0);
  assert.equal(overflowing > 0, name === 'overflow');
  // Exercise both fast and overflowing paths against all-pairs viscosity, not
  // against another invocation of the cached traversal.
  sim.swap();
  encoder = device.createCommandEncoder();
  pass = encoder.beginComputePass();
  sim.run(pass, 'viscosity', params);
  pass.end();
  device.queue.submit([encoder.finish()]);
  const viscous = await read(sim.spare),
    coefficient = (0.002 + 0.025 * 0.065) * PHYSICS_DT * 60;
  for (let i = 0; i < n; i++) {
    const delta = [0, 0, 0];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const r = Math.hypot(
        ...[0, 1, 2].map((k) => prepared[i * 12 + k] - prepared[j * 12 + k]),
      );
      if (r >= h || r < 1e-6) continue;
      const weight = (1 - r / h) ** 2;
      for (let k = 0; k < 3; k++)
        delta[k] +=
          (prepared[j * 12 + 8 + k] - prepared[i * 12 + 8 + k]) * weight;
    }
    for (let k = 0; k < 3; k++)
      assert.ok(
        Math.abs(
          viscous[i * 12 + 8 + k] -
            (prepared[i * 12 + 8 + k] + delta[k] * coefficient),
        ) < 2e-5,
        'overflow fallback must retain every interaction',
      );
  }

  // Pressure positions change after every correction. Poison the old cache,
  // then compare two consecutive pressure rounds to an independent all-pairs
  // oracle. The second round must refresh its list without a grid rebuild.
  let pressureState = prepared;
  for (let round = 0; round < 2; round++) {
    device.queue.writeBuffer(
      sim.starts,
      NEIGHBOR_CACHE_BASE * 4,
      new Uint32Array(n).fill(0),
    );
    encoder = device.createCommandEncoder();
    pass = encoder.beginComputePass();
    sim.run(pass, 'lambda', params);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const pressures = await read(sim.lambda);
    const pressureCache = new Uint32Array((await read(sim.starts)).buffer);
    const expectedNeighbors = [];
    for (let i = 0; i < n; i++) {
      const list = [],
        grad = [0, 0, 0];
      let rho = 0,
        sum = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const diff = [0, 1, 2].map(
          (k) => pressureState[i * 12 + k] - pressureState[j * 12 + k],
        );
        const r = Math.hypot(...diff);
        if (r >= h || r < 1e-6) continue;
        const q = 1 - r / h;
        const gradient = diff.map((x) => (2 * q * x) / (h * 3.6 * r));
        list.push({ j, gradient });
        rho += q * q;
        for (let k = 0; k < 3; k++) grad[k] += gradient[k];
        sum += gradient.reduce((s, x) => s + x * x, 0);
      }
      expectedNeighbors.push(list);
      const lambda =
        (-0.25 * Math.max(rho / 3.6 - 1, 0)) /
        (sum + grad.reduce((s, x) => s + x * x, 0) + 2);
      assert.ok(
        Math.abs(pressures[i * 4] - lambda) < 2e-5,
        'all-pairs pressure multiplier',
      );
      assert.equal(
        pressureCache[NEIGHBOR_CACHE_BASE + i],
        list.length,
        'pressure must refresh poisoned neighbor counts',
      );
      if (list.length <= NEIGHBOR_CACHE_SIZE) {
        const indices = list.map(({ j }) => j);
        const cached = indices.map(
          (_, entry) =>
            pressureCache[
              NEIGHBOR_CACHE_BASE + CAPACITY + entry * CAPACITY + i
            ],
        );
        assert.deepEqual(
          cached.sort((a, b) => a - b),
          indices,
          'fresh pressure neighbor set',
        );
      }
    }
    encoder = device.createCommandEncoder();
    pass = encoder.beginComputePass();
    sim.run(pass, 'correct', params, { 4: sim.lambda, 5: sim.factor });
    pass.end();
    device.queue.submit([encoder.finish()]);
    const corrected = await read(sim.spare);
    for (let i = 0; i < n; i++) {
      const delta = [0, 0, 0];
      for (const { j, gradient } of expectedNeighbors[i])
        for (let k = 0; k < 3; k++)
          delta[k] -= (pressures[i * 4] + pressures[j * 4]) * gradient[k];
      const limiter = Math.min(
        1,
        (0.017 * Math.cbrt(0.2)) / Math.max(Math.hypot(...delta), 1e-8),
      );
      for (let k = 0; k < 3; k++)
        assert.ok(
          Math.abs(
            corrected[i * 12 + k] -
              (pressureState[i * 12 + k] + delta[k] * limiter),
          ) < 2e-5,
          'cached and overflow pressure corrections must match all pairs',
        );
      for (let k = 3; k < 12; k++)
        assert.equal(corrected[i * 12 + k], pressureState[i * 12 + k]);
    }
    pressureState = corrected;
    sim.swap();
  }
  console.log({ name, particles: n, overflowing, empty });
}
assert.deepEqual(errors, []);
console.log(
  'PASS: exact neighbors, fused factors, all-pairs density/viscosity/pressure, cache refresh, empty and overflow fallback',
);
process.exit(0);
