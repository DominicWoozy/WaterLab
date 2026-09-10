// Same-state GPU comparison against the original per-pair/per-voxel formulas.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import {
  WebGPUSimulation,
  RENDER_PARAMETER_SLOT,
} from '../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../app/webgpu/volume.ts';
import { ComputeKernel } from '../app/webgpu/compute.ts';
import { fluidShaders } from '../app/webgpu/fluid-shaders.ts';
import { volumeShaders } from '../app/webgpu/volume-shaders.ts';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const device = await (await gpu.requestAdapter()).requestDevice();
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const sim = await WebGPUSimulation.create(device);
const volume = await WebGPUVolume.create(device);
assert.ok(
  fluidShaders.viscosity.includes(
    'surfacePairWeighted(surface[i],surface[j],diff,r,aux[i].z,aux[j].z,max(aux[i].w,aux[j].w))',
  ),
);
assert.ok(volumeShaders.density.includes('let radius=centre.w;'));
assert.ok(volumeShaders.density.includes('value+=q*q*q*shapes[j].m0.w;'));
const originalViscosity = await ComputeKernel.create(
  device,
  'uncached surface weights',
  fluidShaders.viscosity.replace(
    'surfacePairWeighted(surface[i],surface[j],diff,r,aux[i].z,aux[j].z,max(aux[i].w,aux[j].w))',
    'surfacePair(surface[i],surface[j],diff,r,aux[i].z,aux[j].z)',
  ),
);
const originalDensity = await ComputeKernel.create(
  device,
  'uncached density parameters',
  volumeShaders.density
    .replace(
      'let radius=centre.w;',
      'let radius=mix(.1,.19,smoothstep(.15,1.2,shapes[j].m1.w))*P.clock.z;',
    )
    .replace(
      'value+=q*q*q*shapes[j].m0.w;',
      'value+=q*q*q*(1.+max(0.,1.-shapes[j].m1.w)*.8);',
    ),
);
const cachedDensity = volume.kernels.get('density');
const kept = (globalThis.readbacks = []);
async function read(b, bytes = b.size) {
  const stage = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(b, 0, stage, 0, bytes);
  device.queue.submit([e.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const range = stage.getMappedRange();
  kept.push([stage, range]);
  const result = new Float32Array(range.slice(0));
  stage.unmap();
  return result;
}
function compare(a, b, label, tolerance = 3e-5) {
  assert.equal(a.length, b.length);
  let max = 0,
    different = 0;
  for (let i = 0; i < a.length; i++) {
    const error = Math.abs(a[i] - b[i]);
    assert.ok(
      Number.isFinite(error) &&
        error <= tolerance * Math.max(1, Math.abs(b[i])),
      `${label}[${i}]: ${a[i]} vs ${b[i]}`,
    );
    max = Math.max(max, error);
    if (error > 0) different++;
  }
  return { max, different };
}
const forces = { gravity: 9.8, viscosity: 0.025, agitation: 1.4 };
async function verify(label) {
  const state = await read(sim.state, sim.count * 48);
  const p = sim.writeParameters(RENDER_PARAMETER_SLOT, {
    forces,
    surfaceTension: true,
  });
  let e = device.createCommandEncoder(),
    pass = e.beginComputePass();
  sim.run(pass, 'viscosity', p);
  pass.end();
  device.queue.submit([e.finish()]);
  const fastVelocity = await read(sim.spare, sim.count * 48);
  e = device.createCommandEncoder();
  pass = e.beginComputePass();
  originalViscosity.dispatch(
    pass,
    {
      0: p,
      1: sim.state,
      2: sim.spare,
      3: sim.starts,
      4: sim.factor,
      5: sim.lambda,
      6: sim.duck,
      7: sim.reactions,
      8: sim.surface,
    },
    Math.ceil(sim.count / 128),
  );
  pass.end();
  device.queue.submit([e.finish()]);
  const velocity = compare(
    fastVelocity,
    await read(sim.spare, sim.count * 48),
    label + ' velocity',
    2e-6,
  );
  volume.kernels.set('density', cachedDensity);
  e = device.createCommandEncoder();
  volume.encode(e, sim, true, true);
  device.queue.submit([e.finish()]);
  const raw = await read(volume.density),
    filtered = await read(volume.temp),
    draw = await read(volume.detailDraw);
  volume.kernels.set('density', originalDensity);
  e = device.createCommandEncoder();
  volume.encode(e, sim, true, true);
  device.queue.submit([e.finish()]);
  const density = compare(raw, await read(volume.density), label + ' density');
  const surface = compare(
    filtered,
    await read(volume.temp),
    label + ' filtered',
  );
  assert.deepEqual(await read(volume.detailDraw), draw);
  assert.deepEqual(
    await read(sim.state, sim.count * 48),
    state,
    'optimization leaves physical state untouched',
  );
  console.log({ label, velocity, density, surface });
}
for (const quality of [15000, 30000, 50000]) {
  let e = device.createCommandEncoder();
  sim.reset(e, quality);
  device.queue.submit([e.finish()]);
  for (let tick = 0; tick < 20; tick++) {
    e = device.createCommandEncoder();
    sim.step(e, { forces, splash: tick === 0 ? [0.4, 0.2, 1.5] : undefined });
    device.queue.submit([e.finish()]);
    if (tick % 2 === 1) await device.queue.onSubmittedWorkDone();
  }
  await verify(String(quality));
}
// A dense input exceeds the 96-entry neighbor cache; isolated input exercises
// sub-voxel drops and the low-density radius/amplitude branches.
for (const spacing of [0.006, 0.071]) {
  sim.count = 513;
  const data = new Float32Array(sim.count * 12);
  for (let i = 0; i < sim.count; i++) {
    const p =
      i === 512
        ? [1.5, 2, 1]
        : [
            ((i % 8) - 3.5) * spacing,
            0.7 + ((Math.floor(i / 8) % 8) - 3.5) * spacing,
            (Math.floor(i / 64) - 3.5) * spacing,
          ];
    const v = [Math.sin(i) * 0.01, Math.cos(i) * 0.01, Math.sin(i * 3) * 0.01];
    data.set([...p, 1, ...p.map((x, k) => x - v[k] / 180), 1, ...v, i], i * 12);
  }
  device.queue.writeBuffer(sim.state, 0, data);
  const p = sim.writeParameters(RENDER_PARAMETER_SLOT, {
    forces,
    surfaceTension: true,
  });
  const e = device.createCommandEncoder();
  sim.buildGrid(e, p);
  const pass = e.beginComputePass();
  sim.run(pass, 'prepareVelocity', p, { 5: sim.factor });
  pass.end();
  sim.swap();
  device.queue.submit([e.finish()]);
  await verify(spacing === 0.006 ? 'overflow' : 'isolated');
}
sim.count = 0;
const e = device.createCommandEncoder();
volume.encode(e, sim, true);
device.queue.submit([e.finish()]);
assert.ok(
  (await read(volume.density)).every((x) => x === 0),
  'empty rebuild clears all old density',
);
assert.deepEqual(errors, []);
console.log(
  'PASS cached surface weights and density constants match original formulas, all qualities, overflow and empty',
);
process.exit(0);
