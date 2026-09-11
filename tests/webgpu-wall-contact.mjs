// Exercise the actual filtered particle field, not an analytic full tank.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { WebGPUSimulation } from '../app/webgpu/simulation.ts';
import { WebGPUVolume } from '../app/webgpu/volume.ts';
import { renderScene } from '../app/webgpu/render-shaders.ts';
Object.assign(globalThis, globals);
globalThis.nativeGPU = create(['backend=metal']);
const adapter = await nativeGPU.requestAdapter();
assert.ok(adapter, 'Metal adapter required');
const filtered =
  !process.env.UNFILTERED && adapter.features.has('float32-filterable');
const device = await adapter.requestDevice({
  requiredFeatures: filtered ? ['float32-filterable'] : [],
});
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const sim = await WebGPUSimulation.create(device);
const volume = await WebGPUVolume.create(device);
const input = { forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 } };
const shader = renderScene(filtered);
const original = shader.replace(
  /var sample=p;sample.y=max\(sample.y,-\.917\);\s*if\(p.y<\.8\)\{sample.x=clamp\(sample.x,-1.78,1.78\);sample.z=clamp\(sample.z,-1.28,1.28\);\}/,
  'let sample=p;',
);
assert.notEqual(original, shader, 'baseline must remove boundary continuation');
const entries = [
  {
    binding: 1,
    visibility: GPUShaderStage.COMPUTE,
    texture: {
      viewDimension: '3d',
      sampleType: filtered ? 'float' : 'unfilterable-float',
    },
  },
  {
    binding: 3,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: 'read-only-storage' },
  },
  {
    binding: 15,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: 'read-only-storage' },
  },
  {
    binding: 16,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: 'storage' },
  },
];
if (filtered)
  entries.push({
    binding: 2,
    visibility: GPUShaderStage.COMPUTE,
    sampler: { type: 'filtering' },
  });
const layout = device.createBindGroupLayout({ entries });
const pipelines = [];
for (const source of [original, shader]) {
  const module = device.createShaderModule({
    code:
      source +
      `
@group(0) @binding(15) var<storage,read> queries:array<vec4f>;
@group(0) @binding(16) var<storage,read_write> results:array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 volumeTop=bitcast<f32>(bounds[0])-2.+.36;
 let p=queries[id.x].xyz;results[id.x]=vec4f(density(p),normalAt(p));
}`,
  });
  assert.deepEqual(
    (await module.getCompilationInfo()).messages.filter(
      (m) => m.type === 'error',
    ),
    [],
  );
  pipelines.push(
    await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: 'main' },
    }),
  );
}
const query = device.createBuffer({
  size: 4096,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const result = device.createBuffer({
  size: 4096,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
const staging = device.createBuffer({
  size: 4096,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const resources = [
  { binding: 1, resource: volume.view },
  { binding: 3, resource: { buffer: volume.bounds } },
  { binding: 15, resource: { buffer: query } },
  { binding: 16, resource: { buffer: result } },
];
if (filtered)
  resources.push({
    binding: 2,
    resource: device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
    }),
  });
const group = device.createBindGroup({ layout, entries: resources });
async function sample(points, variant = 1) {
  device.queue.writeBuffer(
    query,
    0,
    new Float32Array(points.flatMap((p) => [...p, 0])),
  );
  const e = device.createCommandEncoder(),
    pass = e.beginComputePass();
  pass.setPipeline(pipelines[variant]);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(points.length);
  pass.end();
  e.copyBufferToBuffer(result, 0, staging, 0, points.length * 16);
  device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  return points.map((_, i) => data[i * 4]);
}
const wet = [
  [1.859, -0.83, 0],
  [-1.859, -0.83, 0],
  [0, -0.83, 1.359],
  [0, -0.83, -1.359],
  [0, -0.959, 0],
  ...[1, -1].flatMap((x) => [1, -1].map((z) => [x * 1.859, -0.959, z * 1.359])),
];
const interior = [
  [0, -0.8, 0],
  [0.6, -0.8, 0.6],
  [1.6, -0.8, 1.1],
  [0, 0.1, 0],
];
for (const quality of [50000, 70000, 100000]) {
  let e = device.createCommandEncoder();
  sim.reset(e, quality);
  device.queue.submit([e.finish()]);
  // The actual solver produces the wet wall; reconstruction must not depend
  // on coarse particles extending far past its collision plane.
  for (let i = 0; i < 90; i++) {
    e = device.createCommandEncoder();
    sim.step(e, input);
    device.queue.submit([e.finish()]);
    if (i % 15 === 14) await device.queue.onSubmittedWorkDone();
  }
  e = device.createCommandEncoder();
  volume.encode(e, sim, false);
  device.queue.submit([e.finish()]);
  const before = await sample(wet, 0),
    after = await sample(wet);
  console.log({ quality, filtered, before, after });
  for (const value of after)
    assert.ok(
      value > 1.15,
      'settled wet wall/floor/corner must reach display boundary',
    );
  assert.deepEqual(
    await sample(interior),
    await sample(interior, 0),
    'interior field unchanged',
  );
  assert.deepEqual(
    await sample([
      [1.861, -0.83, 0],
      [0, -0.961, 0],
      [0, -0.83, 1.361],
    ]),
    [0, 0, 0],
    'no density outside clipping planes',
  );
  // A detached small fluid block near the wall must leave its air gap intact.
  // Repeat above the rim, where there is no side wall to continue toward.
  for (const airborne of [false, true]) {
    const points = [],
      scale = Math.cbrt(10000 / quality),
      spacing = 0.05 * scale;
    for (let x = 0; x < 5; x++)
      for (let y = 0; y < 5; y++)
        for (let z = 0; z < 5; z++)
          points.push([
            (airborne ? 1.78 : 1.6) - x * spacing,
            (airborne ? 1.2 : -0.6) + y * spacing,
            (z - 2) * spacing,
          ]);
    sim.count = points.length;
    const data = new Float32Array(sim.count * 12);
    points.forEach((p, i) => data.set([...p, 1, ...p, 1, 0, 0, 0, i], i * 12));
    device.queue.writeBuffer(sim.state, 0, data);
    e = device.createCommandEncoder();
    volume.encode(e, sim, false);
    device.queue.submit([e.finish()]);
    const p = [[1.859, airborne ? 1.24 : -0.56, 0]];
    if (!airborne)
      assert.equal(
        (await sample(p))[0],
        0,
        'detached block cannot wet the wall',
      );
    assert.equal(
      (await sample([[1.6, -0.959, 0]]))[0],
      0,
      'detached block cannot wet the floor',
    );
    if (airborne)
      assert.deepEqual(
        await sample([
          [1.82, 1.24, 0],
          [1.85, 1.24, 0],
        ]),
        await sample(
          [
            [1.82, 1.24, 0],
            [1.85, 1.24, 0],
          ],
          0,
        ),
        'above-rim reconstruction unchanged',
      );
  }
}
assert.deepEqual(errors, []);
sim.destroy();
volume.destroy();
device.destroy();
console.log('wall contact regression passed');
process.exit(0);
