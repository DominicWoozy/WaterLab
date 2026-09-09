// Native Metal end-to-end benchmark; browser presentation/compositor is excluded.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../app/webgpu/volume.ts';
import { WebGPURenderer } from '../app/webgpu/renderer.ts';
import { readFile, writeFile } from 'node:fs/promises';
import { deflateSync } from 'node:zlib';
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
const volume = await WebGPUVolume.create(device);
const renderer = await WebGPURenderer.create(device, 'rgba8unorm');
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
const paired = process.env.PAIRED === '1';
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
  ? device.createQuerySet({ type: 'timestamp', count: frames * 2 })
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
                beginningOfPassWriteIndex: frame * 2,
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
                    endOfPassWriteIndex: frame * 2 + 1,
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
  frameModes.push(
    process.env.REUSE_GRID === 'alternate'
      ? reuseGrid
      : process.env.CAPILLARY === 'alternate'
        ? implicit
        : process.env.TENSION === 'alternate'
          ? tensionMode
          : detailMode,
  );
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
let gpuTiming;
if (timing) {
  const resolve = device.createBuffer({
    size: frames * 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const mapped = device.createBuffer({
    size: frames * 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.resolveQuerySet(timing, 0, frames * 2, resolve, 0);
  encoder.copyBufferToBuffer(resolve, 0, mapped, 0, frames * 16);
  device.queue.submit([encoder.finish()]);
  await mapped.mapAsync(GPUMapMode.READ);
  const range = mapped.getMappedRange();
  globalThis.timestampReadback = [resolve, mapped, range];
  const values = new BigUint64Array(range.slice(0));
  mapped.unmap();
  const modes = { on: [], off: [] };
  for (let frame = 20; frame < frames; frame++)
    modes[frameModes[frame] ? 'on' : 'off'].push(
      Number(values[frame * 2 + 1] - values[frame * 2]) / 1e6,
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
const stride = Math.ceil((w * 4) / 256) * 256;
const output = device.createBuffer({
  size: stride * h,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
const densityCheck = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
e = device.createCommandEncoder();
for (const [i, z] of [12, 48, 84].entries())
  e.copyBufferToBuffer(
    volume.density,
    (64 + 128 * (11 + 160 * z)) * 4,
    densityCheck,
    i * 4,
    4,
  );
e.copyBufferToBuffer(featureCount, 4, densityCheck, 12, 4);
e.copyTextureToBuffer(
  { texture: image },
  { buffer: output, bytesPerRow: stride },
  [w, h],
);
device.queue.submit([e.finish()]);
await output.mapAsync(GPUMapMode.READ);
// Keep mapped wrappers alive until exit to avoid a native Dawn Node finalizer bug.
const range = output.getMappedRange();
const pixels = new Uint8Array(range.slice(0));
output.unmap();
await densityCheck.mapAsync(GPUMapMode.READ);
const densityRange = densityCheck.getMappedRange();
const densityCopy = densityRange.slice(0);
const densities = new Float32Array(densityCopy);
densityCheck.unmap();
assert.ok(
  densities.slice(0, 3).every((v) => (rough ? v >= 0 : v > 1.15)),
  'Water must cover front, middle and back slices',
);
assert.ok(
  pixels.some((v, i) => i % 4 !== 3 && v > 80),
  'Rendered frame must contain scene colors',
);
assert.ok(
  pixels.some((v, i) => i % 4 !== 3 && v < 50),
  'Rendered frame must contain scene contrast',
);
assert.deepEqual(errors, []);
assert.equal(lost, undefined, JSON.stringify(lost));
const crc32 = (buf) => {
  let c = 0xffffffff;
  for (const v of buf) {
    c ^= v;
    for (let i = 0; i < 8; i++) c = (c >>> 1) ^ (c & 1 ? 0xedb88320 : 0);
  }
  return (c ^ 0xffffffff) >>> 0;
};
const chunk = (name, body) => {
  const tag = Buffer.from(name),
    len = Buffer.alloc(4),
    crc = Buffer.alloc(4);
  len.writeUInt32BE(body.length);
  crc.writeUInt32BE(crc32(Buffer.concat([tag, body])));
  return Buffer.concat([len, tag, body, crc]);
};
const header = Buffer.alloc(13);
header.writeUInt32BE(w);
header.writeUInt32BE(h, 4);
header[8] = 8;
header[9] = 6;
const scanlines = Buffer.alloc((w * 4 + 1) * h);
for (let y = 0; y < h; y++)
  scanlines.set(
    pixels.subarray(y * stride, y * stride + w * 4),
    y * (w * 4 + 1) + 1,
  );
const path = `/private/tmp/water-webgpu-${n}.png`;
await writeFile(
  path,
  Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(scanlines)),
    chunk('IEND', Buffer.alloc(0)),
  ]),
);
const sorted = [...pairs].sort((a, b) => a - b);
console.log(
  JSON.stringify(
    {
      particles: n,
      resolution: [w, h],
      features,
      scene: rough ? 'rough' : 'calm',
      paired,
      detailCount: new Uint32Array(densityCopy)[3],
      comparison:
        process.env.REUSE_GRID === 'alternate'
          ? 'reuse final grid and neighbors'
          : process.env.CAPILLARY === 'alternate'
            ? 'implicit (on) / explicit (off) capillary'
            : process.env.TENSION === 'alternate'
              ? 'surface tension'
              : 'detail reconstruction',
      modes: Object.fromEntries(
        Object.entries(byMode)
          .filter(([, a]) => a.length)
          .map(([k, a]) => {
            const sorted = [...a].sort((x, y) => x - y);
            return [
              k,
              {
                mean: a.reduce((a, b) => a + b) / a.length,
                p50: sorted[Math.floor(a.length * 0.5)],
                p95: sorted[Math.floor(a.length * 0.95)],
              },
            ];
          }),
      ),
      gpu_command_span_ms: gpuTiming,
      completed_frame_ms: {
        mean: pairs.reduce((a, b) => a + b) / pairs.length,
        p50: sorted[Math.floor(sorted.length * 0.5)],
        p95: sorted[Math.floor(sorted.length * 0.95)],
      },
      image: path,
      split,
      synchronized_phase_ms: profile
        ? Object.fromEntries(
            Object.entries(phases).map(([k, v]) => [
              k,
              v.reduce((a, b) => a + b) / v.length,
            ]),
          )
        : undefined,
      errors,
    },
    null,
    2,
  ),
);
await device.queue.onSubmittedWorkDone();
process.exit(0);
