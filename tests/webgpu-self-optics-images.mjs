// Frozen production images: independently exercise optional self optics.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../app/webgpu/volume.ts';
import { WebGPURenderer } from '../app/webgpu/renderer.ts';
import { renderShader } from '../app/webgpu/render-shaders.ts';
Object.assign(globalThis, globals);
globalThis.nativeGPU = create(['backend=metal']);
const adapter = await nativeGPU.requestAdapter();
const filtered =
  !process.env.UNFILTERED && adapter.features.has('float32-filterable');
const device = await adapter.requestDevice({
  requiredFeatures: filtered ? ['float32-filterable'] : [],
});
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const volume = await WebGPUVolume.create(device),
  sim = await WebGPUSimulation.create(device);
const renderer = await WebGPURenderer.create(device, 'rgba8unorm');
const bytes = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
renderer.setModel(
  bytes(await readFile('public/models/duck/bvh.bin')),
  bytes(await readFile('public/models/duck/triangles.bin')),
);
device.queue.writeBuffer(
  sim.duck,
  0,
  new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
);
const pipelines = await Promise.all(
  [renderShader(filtered), renderShader(filtered, true)].map(async (code) => {
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    assert.deepEqual(
      info.messages.filter((x) => x.type === 'error'),
      [],
    );
    return device.createRenderPipelineAsync({
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
    });
  }),
);
const w = 256,
  h = 192;
const scratch = device.createTexture({
  size: [w, h],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const target = device.createTexture({
  size: [w, h],
  format: 'rgba32float',
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const depth = device.createTexture({
  size: [w, h],
  format: 'depth32float',
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const kept = (globalThis.kept = []);
const field = new Float32Array(128 * 160 * 96);
for (let z = 0; z < 96; z++)
  for (let y = 0; y < 160; y++)
    for (let x = 0; x < 128; x++) {
      const px = -2.08 + (x / 127) * 4.16,
        py = -1.12 + (y / 159) * 5.2,
        pz = -1.56 + (z / 95) * 3.12;
      const water = py < -0.1 + 0.12 * Math.sin(px * 4) * Math.cos(pz * 3);
      const slab =
        Math.abs(py - 0.65) < 0.07 &&
        px > -1.4 &&
        px < -0.2 &&
        Math.abs(pz) < 0.8;
      const sphere = (px - 0.6) ** 2 + (py - 1.2) ** 2 + pz * pz < 0.22 ** 2;
      field[x + 128 * (y + 160 * z)] = water || slab || sphere ? 4 : 0;
    }
const norm = (a) => {
  const n = Math.hypot(...a);
  return a.map((x) => x / n);
};
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
async function probe(pipeline) {
  const e = device.createCommandEncoder(),
    pass = e.beginRenderPass({
      colorAttachments: [
        { view: target.createView(), loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: {
        view: depth.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1,
      },
    });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, renderer.surfaceGroups.get(sim.duck));
  pass.draw(3);
  pass.end();
  const read = device.createBuffer({
    size: w * h * 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  e.copyTextureToBuffer(
    { texture: target },
    { buffer: read, bytesPerRow: w * 16 },
    [w, h],
  );
  device.queue.submit([e.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const range = read.getMappedRange();
  kept.push([read, range]);
  const values = new Float32Array(range.slice(0));
  read.unmap();
  return values;
}
const results = [];
for (const [name, eye, wet] of [
  ['above', [3, 2.5, 4], true],
  ['side', [3, 0.2, 4], true],
  ['dry', [3, 2.5, 4], false],
]) {
  device.queue.writeBuffer(volume.bounds, 0, new Float32Array([3.64, 0, 0, 0]));
  device.queue.writeTexture(
    { texture: volume.texture },
    wet ? field : new Float32Array(field.length),
    { bytesPerRow: 512, rowsPerImage: 160 },
    [128, 160, 96],
  );
  const forward = norm(eye.map((x) => -x)),
    right = norm(cross(forward, [0, 1, 0])),
    up = cross(right, forward);
  const init = device.createCommandEncoder();
  renderer.encode(
    init,
    scratch.createView(),
    sim,
    volume,
    { eye, forward, right, up },
    w,
    h,
    { light: 1.3, reflection: true, caustics: false, particles: false },
  );
  device.queue.submit([init.finish()]);
  const baseline = await probe(pipelines[0]);
  for (const [mode, reflection, shadow] of [
    ['reflection', 1, 0],
    ['shadow', 0, 1],
    ['both', 1, 1],
  ]) {
    device.queue.writeBuffer(
      renderer.uniform,
      28,
      new Float32Array([reflection]),
    );
    device.queue.writeBuffer(renderer.uniform, 44, new Float32Array([shadow]));
    const values = await probe(pipelines[1]);
    let max = 0,
      squared = 0,
      changed = 0;
    for (let i = 0; i < values.length; i++) {
      assert.ok(Number.isFinite(values[i]), 'finite pixels');
      const delta = Math.abs(values[i] - baseline[i]);
      max = Math.max(max, delta);
      squared += delta * delta;
      if (delta > 1e-5) changed++;
    }
    if (wet)
      assert.ok(changed > 10, `${name} ${mode} must affect visible water`);
    else assert.equal(max, 0, 'dry background unchanged');
    results.push({
      name,
      mode,
      max,
      rms: Math.sqrt(squared / values.length),
      changed,
    });
  }
  device.queue.writeBuffer(renderer.uniform, 28, new Float32Array([0]));
  device.queue.writeBuffer(renderer.uniform, 44, new Float32Array([0]));
  assert.deepEqual(
    await probe(pipelines[0]),
    baseline,
    'disable restores exact original pixels',
  );
}
assert.deepEqual(errors, []);
console.log(JSON.stringify({ filtered, results, errors }));
console.log(
  'PASS: both optional effects change water, preserve dry background, and toggle back exactly',
);
process.exit(0);
