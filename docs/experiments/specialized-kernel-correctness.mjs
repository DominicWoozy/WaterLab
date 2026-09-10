import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { create, globals } from 'webgpu';
import {
  WebGPUSimulation,
  RENDER_PARAMETER_SLOT,
  PHYSICS_DT,
} from '../../app/webgpu/simulation.ts';
import { specializedKernels } from './specialized-kernels.mjs';

// Existing independent oracles also exercise transitions between all modes.
const hook = new URL('./specialized-kernels.mjs', import.meta.url).href;
for (const fixture of ['webgpu-neighbors.mjs', 'webgpu-surface.mjs']) {
  let source = readFileSync(
    new URL(`../../tests/${fixture}`, import.meta.url),
    'utf8',
  );
  assert.ok(source.includes('const kept ='));
  source = source
    .replaceAll("'../app/", `'${new URL('../../app/', import.meta.url).href}`)
    .replace(
      'const kept =',
      `const {specializedKernels}=await import(${JSON.stringify(hook)});\nawait specializedKernels(sim,{prune:true,freeze:process.env.FREEZE==='1'});\nconst kept =`,
    );
  execFileSync(
    process.execPath,
    ['--experimental-strip-types', '--input-type=module', '--eval', source],
    { stdio: 'inherit', cwd: new URL('../../', import.meta.url) },
  );
}

Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const device = await (await gpu.requestAdapter()).requestDevice();
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const sim = await WebGPUSimulation.create(device);
const control = await specializedKernels(sim, {
  prune: true,
  freeze: process.env.FREEZE === '1',
});
const kept = (globalThis.readbacks = []);
async function read(b) {
  const stage = device.createBuffer({
    size: b.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(b, 0, stage, 0, b.size);
  device.queue.submit([e.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const range = stage.getMappedRange();
  kept.push([stage, range]);
  const result = new Float32Array(range.slice(0));
  stage.unmap();
  return result;
}
for (const quality of [15000, 30000, 50000])
  for (const spacing of [0.06, 0.006]) {
    let e = device.createCommandEncoder();
    sim.reset(e, quality);
    device.queue.submit([e.finish()]);
    sim.count = 513;
    const data = new Float32Array(513 * 12);
    for (let i = 0; i < 513; i++) {
      const p =
        i === 512
          ? [1.5, 2, 1]
          : [
              ((i % 8) - 3.5) * spacing,
              0.7 + ((Math.floor(i / 8) % 8) - 3.5) * spacing,
              (Math.floor(i / 64) - 3.5) * spacing,
            ];
      const v = [Math.sin(i) * 0.3, Math.cos(i) * 0.2, Math.sin(3 * i) * 0.1];
      data.set(
        [...p, 1, ...p.map((x, k) => x - v[k] * PHYSICS_DT), 1, ...v, i],
        i * 12,
      );
    }
    device.queue.writeBuffer(sim.state, 0, data);
    device.queue.writeBuffer(
      sim.duck,
      0,
      new Float32Array([0, 10, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    for (const mode of [0, 1, 2, 1, 0]) {
      const p = sim.writeParameters(RENDER_PARAMETER_SLOT, {
        forces: { gravity: 0, viscosity: 0.025, agitation: 0 },
        surfaceTension: mode !== 0,
        capillaryMode: mode === 2 ? 'implicit' : 'explicit',
      });
      e = device.createCommandEncoder();
      sim.buildGrid(e, p);
      let pass = e.beginComputePass();
      sim.run(pass, 'prepareVelocity', p, { 5: sim.factor });
      pass.end();
      device.queue.submit([e.finish()]);
      sim.swap();
      const outputs = [];
      for (const enabled of [false, true]) {
        control.enabled = enabled;
        e = device.createCommandEncoder();
        pass = e.beginComputePass();
        sim.run(pass, 'viscosity', p);
        pass.end();
        device.queue.submit([e.finish()]);
        outputs.push(await read(sim.spare));
      }
      let maxAbs = 0;
      for (let i = 0; i < 513 * 12; i++) {
        assert.ok(Number.isFinite(outputs[1][i]));
        maxAbs = Math.max(maxAbs, Math.abs(outputs[1][i] - outputs[0][i]));
      }
      assert.ok(
        maxAbs < 2e-6,
        `viscosity ${quality}/${spacing}/${mode}: ${maxAbs}`,
      );
      console.log({ quality, spacing, mode, maxAbs });
    }
  }
assert.deepEqual(errors, []);
console.log(
  'PASS specialized shaders: all quality modes, cached/overflow/isolated neighbors, off/explicit/implicit transitions',
);
process.exit(0);
