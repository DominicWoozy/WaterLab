// Native Metal end-to-end benchmark; browser presentation/compositor is excluded.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../../app/webgpu/volume.ts';
import { WebGPURenderer } from '../../app/webgpu/renderer.ts';
import { readFile } from 'node:fs/promises';
Object.assign(globalThis, globals);
const gpu = create(['backend=metal']);
globalThis.nativeGPU = gpu;
const adapter = await gpu.requestAdapter();
const features =
  !process.env.UNFILTERED && adapter.features.has('float32-filterable')
    ? ['float32-filterable']
    : [];
if (adapter.features.has('timestamp-query')) features.push('timestamp-query');
const device = await adapter.requestDevice({ requiredFeatures: features });
const errors = [];
let lost;
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
void device.lost.then((info) => {
  lost = info;
});
const sim = await WebGPUSimulation.create(device);
const currentVolume = await WebGPUVolume.create(device);
const volume = currentVolume;
const renderer = await WebGPURenderer.create(device, 'rgba8unorm');
// Point BASELINE_SHADER at a preserved render-shaders.ts (with its sibling imports).
const { renderShader: baselineShader } = await import(
  process.env.BASELINE_SHADER
);
const module = device.createShaderModule({
  code: baselineShader(features.includes('float32-filterable')),
});
const baseline = await device.createRenderPipelineAsync({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [renderer.surfaceLayout],
  }),
  vertex: { module, entryPoint: 'vertex' },
  fragment: {
    module,
    entryPoint: 'fragment',
    targets: [{ format: 'rgba8unorm' }],
  },
  primitive: { topology: 'triangle-list' },
  depthStencil: {
    format: 'depth32float',
    depthWriteEnabled: true,
    depthCompare: 'always',
  },
});
const updated = renderer.surface;
// Include range-cache construction only in the accelerated frame.
const buildRanges = volume.encodeRanges.bind(volume);
volume.encodeRanges = (encoder) => {
  if (renderer.surface === updated) buildRanges(encoder);
};
const b = await readFile('public/models/duck/bvh.bin'),
  t = await readFile('public/models/duck/triangles.bin');
renderer.setModel(
  b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength),
  t.buffer.slice(t.byteOffset, t.byteOffset + t.byteLength),
);
const n = Number(process.argv[2] || 50000),
  w = 1250,
  h = 800,
  frames =
    process.env.REUSE_GRID === 'alternate' ||
    process.env.DETAILS === 'alternate' ||
    process.env.TENSION === 'alternate' ||
    process.env.CAPILLARY === 'alternate'
      ? 140
      : 80;
const image = device.createTexture({
  size: [w, h],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
let e = device.createCommandEncoder();
sim.reset(e, n);
device.queue.submit([e.finish()]);
const rough = process.env.SCENE === 'rough';
const paired = true;
const forces = { gravity: 9.8, viscosity: 0.025, agitation: 0 };
for (let i = 0; i < 120; i++) {
  e = device.createCommandEncoder();
  sim.step(e, {
    surfaceTension: process.env.TENSION !== '0',
    capillaryMode:
      process.env.CAPILLARY === 'implicit' ? 'implicit' : 'explicit',
    forces: rough ? { ...forces, agitation: 1.4 } : forces,
    shake: rough && i < 30 ? 12 : 0,
    splash: rough && i % 60 === 0 ? [0.5, 0.3, 2] : undefined,
  });
  device.queue.submit([e.finish()]);
  if (i % 2 === 1) await device.queue.onSubmittedWorkDone();
}
const seed = device.createBuffer({
  size: sim.state.size,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const reactionSeed = device.createBuffer({
  size: 32,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const duckSeed = device.createBuffer({
  size: 64,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const featureCount = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const anchorTime = sim.time;
if (paired) {
  e = device.createCommandEncoder();
  e.copyBufferToBuffer(sim.state, 0, seed, 0, sim.state.size);
  e.copyBufferToBuffer(sim.duck, 0, duckSeed, 0, 64);
  e.copyBufferToBuffer(sim.reactionTotal, 0, reactionSeed, 0, 32);
  device.queue.submit([e.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const norm = (a) => {
  const n = Math.hypot(...a);
  return a.map((v) => v / n);
};
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const eye = [
    Math.sin(0.58) * Math.cos(0.49) * 8,
    Math.sin(0.49) * 8,
    Math.cos(0.58) * Math.cos(0.49) * 8,
  ],
  forward = norm([-eye[0], -0.05 - eye[1], -eye[2]]),
  right = norm(cross(forward, [0, 1, 0])),
  up = cross(right, forward);
// GPU timestamps separate command-stream time from host completion scheduling.
// Instrument only the benchmark; production render/compute paths are unchanged.
const timing = features.includes('timestamp-query')
  ? device.createQuerySet({ type: 'timestamp', count: frames * 4 })
  : null;
function timedEncoder(frame) {
  const encoder = device.createCommandEncoder();
  if (!timing) return encoder;
  let started = false;
  return new Proxy(encoder, {
    get(target, key) {
      if (key === 'beginComputePass')
        return (descriptor) => {
          // A tick has several physical substeps; measure from the first one.
          if (descriptor?.label === 'predict' && !started) {
            started = true;
            return target.beginComputePass({
              ...descriptor,
              timestampWrites: {
                querySet: timing,
                beginningOfPassWriteIndex: frame * 4,
              },
            });
          }
          return target.beginComputePass(descriptor);
        };
      if (key === 'beginRenderPass')
        return (descriptor) =>
          target.beginRenderPass(
            descriptor?.label === 'water and duck'
              ? {
                  ...descriptor,
                  timestampWrites: {
                    querySet: timing,
                    beginningOfPassWriteIndex: frame * 4 + 2,
                    endOfPassWriteIndex: frame * 4 + 3,
                  },
                }
              : descriptor,
          );
      const value = Reflect.get(target, key, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
}
const frameModes = [];
const pairs = [];
const byMode = { on: [], off: [] };
const phases = { physics: [], density: [], render: [] };
const profile = process.env.PROFILE === '1';
const split = !profile && process.env.SPLIT !== '0';
let start = performance.now();
for (let frame = 0; frame < frames; frame++) {
  const commands = [];
  e = timedEncoder(frame);
  if (paired) {
    e.copyBufferToBuffer(seed, 0, sim.state, 0, seed.size);
    e.copyBufferToBuffer(duckSeed, 0, sim.duck, 0, 64);
    e.copyBufferToBuffer(reactionSeed, 0, sim.reactionTotal, 0, 32);
    sim.time = anchorTime;
  }
  const tensionMode =
    process.env.TENSION === 'alternate'
      ? Math.floor(frame / (paired ? 2 : 20)) % 2 === 0
      : process.env.TENSION !== '0';
  const implicit =
    process.env.CAPILLARY === 'alternate'
      ? Math.floor(frame / 2) % 2 === 0
      : process.env.CAPILLARY === 'implicit';
  const optimized = Math.floor(frame / 2) % 2 === 0;
  renderer.surface = optimized ? updated : baseline;

  sim.step(e, {
    surfaceTension: tensionMode,
    capillaryMode: implicit ? 'implicit' : 'explicit',
    forces: rough ? { ...forces, agitation: 1.4 } : forces,
    shake: !paired && rough && frame < 20 ? 12 : 0,
    splash: !paired && rough && frame % 40 === 0 ? [0.5, 0.3, 2] : undefined,
  });
  const detailMode =
    process.env.DETAILS === 'alternate'
      ? Math.floor(frame / (paired ? 2 : 20)) % 2 === 0
      : process.env.DETAILS !== '0';
  const reuseGrid =
    process.env.REUSE_GRID === 'alternate'
      ? Math.floor(frame / (paired ? 2 : 20)) % 2 === 0
      : process.env.REUSE_GRID !== '0';
  frameModes.push(optimized);
  let phaseStart = performance.now();
  if (profile) {
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (frame >= 20) phases.physics.push(performance.now() - phaseStart);
    phaseStart = performance.now();
    e = timedEncoder(frame);
  }
  if (split) {
    commands.push(e.finish());
    e = timedEncoder(frame);
  }
  volume.encode(e, sim, detailMode, reuseGrid);
  if (detailMode)
    e.copyBufferToBuffer(volume.detailDraw, 0, featureCount, 0, 16);
  if (profile) {
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (frame >= 20) phases.density.push(performance.now() - phaseStart);
    phaseStart = performance.now();
    e = timedEncoder(frame);
  }
  if (split) {
    commands.push(e.finish());
    e = timedEncoder(frame);
  }
  renderer.encode(
    e,
    image.createView(),
    sim,
    volume,
    { eye, forward, right, up },
    w,
    h,
    { light: 1.3, reflection: true, caustics: true, particles: false },
  );
  device.queue.submit([...commands, e.finish()]);
  if (profile) {
    await device.queue.onSubmittedWorkDone();
    if (frame >= 20) phases.render.push(performance.now() - phaseStart);
  }
  if (frame % 2 === 1) {
    await device.queue.onSubmittedWorkDone();
    const now = performance.now();
    if (frame >= 20) {
      pairs.push((now - start) / 2);
      byMode[frameModes[frame] ? 'on' : 'off'].push((now - start) / 2);
    }
    start = now;
  }
  assert.equal(lost, undefined, JSON.stringify(lost));
}
let gpuTiming, surfaceTiming;
if (timing) {
  const resolve = device.createBuffer({
    size: frames * 32,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const mapped = device.createBuffer({
    size: frames * 32,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.resolveQuerySet(timing, 0, frames * 4, resolve, 0);
  encoder.copyBufferToBuffer(resolve, 0, mapped, 0, frames * 32);
  device.queue.submit([encoder.finish()]);
  await mapped.mapAsync(GPUMapMode.READ);
  const range = mapped.getMappedRange();
  globalThis.timestampReadback = [resolve, mapped, range];
  const values = new BigUint64Array(range.slice(0));
  mapped.unmap();
  const modes = { on: [], off: [] },
    surfaces = { on: [], off: [] };
  for (let frame = 20; frame < frames; frame++) {
    modes[frameModes[frame] ? 'on' : 'off'].push(
      Number(values[frame * 4 + 3] - values[frame * 4]) / 1e6,
    );
    surfaces[frameModes[frame] ? 'on' : 'off'].push(
      Number(values[frame * 4 + 3] - values[frame * 4 + 2]) / 1e6,
    );
  }
  surfaceTiming = Object.fromEntries(
    Object.entries(surfaces).map(([mode, a]) => [
      mode,
      {
        mean: a.reduce((x, y) => x + y, 0) / a.length,
        p50: a.sort((x, y) => x - y)[Math.floor(a.length / 2)],
      },
    ]),
  );
  gpuTiming = Object.fromEntries(
    Object.entries(modes)
      .filter(([, a]) => a.length)
      .map(([mode, a]) => {
        a.sort((x, y) => x - y);
        return [
          mode,
          {
            mean: a.reduce((x, y) => x + y, 0) / a.length,
            p50: a[Math.floor(a.length * 0.5)],
            p95: a[Math.floor(a.length * 0.95)],
          },
        ];
      }),
  );
}

assert.deepEqual(errors, []);
assert.equal(lost, undefined);
console.log(
  JSON.stringify(
    {
      comparison:
        'conservative macrocell skipping including cache build (on = accelerated)',
      features,
      particles: n,
      scene: rough ? 'rough' : 'calm',
      resolution: [w, h],
      split,
      gpu_command_span_ms: gpuTiming,
      surface_pass_ms: surfaceTiming,
      errors,
    },
    null,
    2,
  ),
);
await device.queue.onSubmittedWorkDone();
process.exit(0);
