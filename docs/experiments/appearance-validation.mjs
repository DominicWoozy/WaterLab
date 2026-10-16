import assert from 'node:assert/strict';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { deflateSync } from 'node:zlib';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../../app/webgpu/volume.ts';
import { WebGPURenderer } from '../../app/webgpu/renderer.ts';
import { appearanceShader } from './appearance-variants.mjs';
Object.assign(globalThis, globals);
const gpu = (globalThis.nativeGPU = create(['backend=metal']));
const adapter = await gpu.requestAdapter();
const filtered = adapter.features.has('float32-filterable');
const device = await adapter.requestDevice({
  requiredFeatures: [
    ...(filtered ? ['float32-filterable'] : []),
    'timestamp-query',
  ],
});
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const sim = await WebGPUSimulation.create(device),
  volume = await WebGPUVolume.create(device),
  renderer = await WebGPURenderer.create(device, 'rgba8unorm');
const bytes = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
renderer.setModel(
  bytes(await readFile('public/models/duck/bvh.bin')),
  bytes(await readFile('public/models/duck/triangles.bin')),
);
let encoder = device.createCommandEncoder();
sim.reset(encoder, 50000);
device.queue.submit([encoder.finish()]);
for (let i = 0; i < 90; i++) {
  encoder = device.createCommandEncoder();
  sim.step(encoder, {
    forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 },
    shake: i < 20 ? 10 : 0,
    splash: i === 70 ? [0.5, 0.3, 2] : undefined,
  });
  device.queue.submit([encoder.finish()]);
  if (i % 2 === 1) await device.queue.onSubmittedWorkDone();
}
encoder = device.createCommandEncoder();
volume.encode(encoder, sim, true, true);
device.queue.submit([encoder.finish()]);
await device.queue.onSubmittedWorkDone();
const norm = (a) => {
    let n = Math.hypot(...a);
    return a.map((x) => x / n);
  },
  cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
const w = 512,
  h = 320;
let camera;
function view(eye) {
  const forward = norm([-eye[0], -0.05 - eye[1], -eye[2]]),
    right = norm(cross(forward, [0, 1, 0]));
  return { eye, forward, right, up: cross(right, forward) };
}
camera = view([3, 2.5, 4]);
const modes = ['baseline', 'aa', 'srgb'];
const pipelines = {};
for (const mode of modes)
  for (const highlights of mode === 'srgb' ? [false] : [false, true]) {
    const module = device.createShaderModule({
      code: appearanceShader(filtered, mode, highlights),
    });
    const info = await module.getCompilationInfo();
    assert.deepEqual(
      info.messages.filter((x) => x.type === 'error'),
      [],
    );
    pipelines[`${mode}-${highlights}`] = await device.createRenderPipelineAsync(
      {
        layout: device.createPipelineLayout({
          bindGroupLayouts: [renderer.surfaceLayout],
        }),
        vertex: { module, entryPoint: 'vertex' },
        fragment: {
          module,
          entryPoint: 'fragment',
          targets: [{ format: 'rgba32float' }],
        },
        depthStencil: {
          format: 'depth32float',
          depthWriteEnabled: true,
          depthCompare: 'always',
        },
      },
    );
  }
const query = device.createQuerySet({ type: 'timestamp', count: 240 });
let frame = 0;
const images = {};
async function render(
  mode,
  scale = 1,
  highlights = false,
  read = true,
  timed = false,
) {
  const width = w * scale,
    height = h * scale;
  const scratch = device.createTexture({
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  // Populate production lighting and detail silhouettes at this exact resolution.
  const init = device.createCommandEncoder();
  renderer.encode(
    init,
    scratch.createView(),
    sim,
    volume,
    camera,
    width,
    height,
    {
      light: 1.3,
      reflection: true,
      caustics: true,
      particles: false,
      lightingRevision: 1,
    },
  );
  device.queue.submit([init.finish()]);
  await device.queue.onSubmittedWorkDone();
  scratch.destroy();
  const image = device.createTexture({
    size: [width, height],
    format: 'rgba32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const depth = device.createTexture({
    size: [width, height],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      { view: image.createView(), loadOp: 'clear', storeOp: 'store' },
    ],
    depthStencilAttachment: {
      view: depth.createView(),
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
      depthClearValue: 1,
    },
    ...(timed
      ? {
          timestampWrites: {
            querySet: query,
            beginningOfPassWriteIndex: frame * 2,
            endOfPassWriteIndex: frame * 2 + 1,
          },
        }
      : {}),
  });
  pass.setPipeline(pipelines[`${mode}-${highlights}`]);
  pass.setBindGroup(0, renderer.surfaceGroups.get(sim.duck));
  pass.draw(3);
  pass.end();
  const stage = read
    ? device.createBuffer({
        size: width * height * 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
    : null;
  if (stage)
    encoder.copyTextureToBuffer(
      { texture: image },
      { buffer: stage, bytesPerRow: width * 16 },
      [width, height],
    );
  device.queue.submit([encoder.finish()]);
  let result;
  if (stage) {
    await stage.mapAsync(GPUMapMode.READ);
    result = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    assert.ok(result.every(Number.isFinite));
  } else await device.queue.onSubmittedWorkDone();
  image.destroy();
  depth.destroy();
  return result;
}
// Plain PNG writer so experiment artifacts don't require an image dependency.
const crc = (b) => {
  let c = ~0;
  for (const v of b) {
    c ^= v;
    for (let j = 0; j < 8; j++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  return ~c >>> 0;
};
function png(data, width, height) {
  const chunk = (type, data) => {
    const b = Buffer.concat([Buffer.from(type), data]),
      n = Buffer.alloc(4),
      c = Buffer.alloc(4);
    n.writeUInt32BE(data.length);
    c.writeUInt32BE(crc(b));
    return Buffer.concat([n, b, c]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 6;
  const raw = Buffer.alloc(height * (width * 4 + 1));
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width * 4; x++)
      raw[y * (width * 4 + 1) + 1 + x] = Math.round(
        Math.min(1, Math.max(0, data[y * width * 4 + x])) * 255,
      );
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}
const out = '/tmp/water-appearance-validation';
await mkdir(out, { recursive: true });
const results = [];
for (const [name, eye] of [
  ['above', [3, 2.5, 4]],
  ['side', [3, 0.2, 4]],
  ['far', [5, 4, 7]],
]) {
  camera = view(eye);
  for (const mode of modes) {
    images[mode] = await render(mode);
    await writeFile(`${out}/${name}-${mode}.png`, png(images[mode], w, h));
  }
  const reference = await render('baseline', 4, true),
    base = await render('baseline', 1, true),
    aa = await render('aa', 1, true);
  let baseError = 0,
    aaError = 0,
    energy = 0,
    baseEnergy = 0,
    aaEnergy = 0,
    n = 0,
    delta = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let ref = 0;
      for (let j = 0; j < 4; j++)
        for (let i = 0; i < 4; i++)
          ref += reference[((y * 4 + j) * w * 4 + x * 4 + i) * 4] / 16;
      const index = (y * w + x) * 4;
      const a = base[index],
        b = aa[index];
      if (ref > 1e-5 || a > 1e-5 || b > 1e-5) {
        baseError += (a - ref) ** 2;
        aaError += (b - ref) ** 2;
        n++;
        energy += ref;
        baseEnergy += a;
        aaEnergy += b;
      }
      for (let k = 0; k < 3; k++)
        delta += Math.abs(images.baseline[index + k] - images.srgb[index + k]);
    }
  results.push({
    name,
    highlightPixels: n,
    baselineRMSE: Math.sqrt(baseError / n),
    aaRMSE: Math.sqrt(aaError / n),
    referenceEnergy: energy,
    baselineEnergy: baseEnergy,
    aaEnergy,
    colorMeanAbsoluteDelta: delta / (w * h * 3),
  });
}
camera = view([3, 2.5, 4]);
const timings = { baseline: [], aa: [], srgb: [] };
for (frame = 0; frame < 120; frame++)
  await render(modes[frame % 3], 1, false, false, true);
const resolve = device.createBuffer({
    size: 1920,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  }),
  stage = device.createBuffer({
    size: 1920,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
encoder = device.createCommandEncoder();
encoder.resolveQuerySet(query, 0, 240, resolve, 0);
encoder.copyBufferToBuffer(resolve, 0, stage, 0, 1920);
device.queue.submit([encoder.finish()]);
await stage.mapAsync(GPUMapMode.READ);
const times = new BigUint64Array(stage.getMappedRange().slice(0));
stage.unmap();
for (let f = 12; f < 120; f++)
  timings[modes[f % 3]].push(Number(times[f * 2 + 1] - times[f * 2]) / 1e6);
const report = {
  filtered,
  resolution: [w, h],
  results,
  surface_gpu_ms: Object.fromEntries(
    Object.entries(timings).map(([k, v]) => [
      k,
      v.reduce((a, b) => a + b, 0) / v.length,
    ]),
  ),
  images: out,
  errors,
};
await writeFile(`${out}/results.json`, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
assert.deepEqual(errors, []);
process.exit(0);
