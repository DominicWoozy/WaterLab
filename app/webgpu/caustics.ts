import { buffer, ComputeKernel } from './compute.ts';
import {
  CAUSTIC_GRID,
  CAUSTIC_MAP,
  causticTraceShader,
  causticDepositShader,
  duckLightShader,
} from './caustic-shaders.ts';
import type { WebGPUVolume } from './volume.ts';

export class WebGPUCaustics {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly photons: GPUBuffer;
  readonly duckDraw: GPUBuffer;
  duckLighting: GPUBuffer;
  private duckVertices = 0;
  private duckPhotons: GPUBuffer;
  private duckResolve!: ComputeKernel;
  private trace!: GPUComputePipeline;
  private deposit!: GPURenderPipeline;
  private traceLayout: GPUBindGroupLayout;
  private depositLayout: GPUBindGroupLayout;
  private depositGroup: GPUBindGroup;
  private groups = new Map<GPUBuffer, GPUBindGroup>();
  private device: GPUDevice;
  private uniform: GPUBuffer;
  private control: GPUBuffer;
  private constructor(device: GPUDevice, uniform: GPUBuffer) {
    this.device = device;
    this.uniform = uniform;
    this.control = buffer(
      device,
      'light transport settings',
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.photons = buffer(
      device,
      'refracted photon wavefront',
      CAUSTIC_GRID[0] * CAUSTIC_GRID[1] * 64,
    );
    this.duckPhotons = buffer(
      device,
      'compact duck photon indices',
      CAUSTIC_GRID[0] * CAUSTIC_GRID[1] * 4,
    );
    this.duckDraw = buffer(device, 'duck photon count', 16);
    this.duckLighting = buffer(device, 'duck surface irradiance', 16);
    this.texture = device.createTexture({
      label: 'receiver direct and refracted irradiance',
      size: [...CAUSTIC_MAP],
      format: 'rgba16float',
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    this.view = this.texture.createView();
    const C = GPUShaderStage.COMPUTE,
      filtered = device.features.has('float32-filterable');
    this.traceLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        {
          binding: 1,
          visibility: C,
          texture: {
            viewDimension: '3d',
            sampleType: filtered ? 'float' : 'unfilterable-float',
          },
        },
        {
          binding: 2,
          visibility: C,
          sampler: { type: filtered ? 'filtering' : 'non-filtering' },
        },
        ...[3, 4, 5, 6].map((binding) => ({
          binding,
          visibility: C,
          buffer: { type: 'read-only-storage' as const },
        })),
        ...[12, 17, 18].map((binding) => ({
          binding,
          visibility: C,
          buffer: { type: 'storage' as const },
        })),
        { binding: 16, visibility: C, buffer: { type: 'uniform' } },
      ],
    });
    this.depositLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 12,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage' },
        },
      ],
    });
    this.depositGroup = device.createBindGroup({
      layout: this.depositLayout,
      entries: [{ binding: 12, resource: { buffer: this.photons } }],
    });
  }
  static async create(device: GPUDevice, uniform: GPUBuffer) {
    const c = new WebGPUCaustics(device, uniform);
    try {
      const trace = device.createShaderModule({
        label: 'caustic surface rays',
        code: causticTraceShader(device.features.has('float32-filterable')),
      });
      const deposit = device.createShaderModule({
        label: 'caustic wavefront deposition',
        code: causticDepositShader,
      });
      for (const shader of [trace, deposit]) {
        const errors = (await shader.getCompilationInfo()).messages.filter(
          (m) => m.type === 'error',
        );
        if (errors.length)
          throw new Error(
            errors.map((m) => `${m.lineNum}: ${m.message}`).join('\n'),
          );
      }
      c.trace = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [c.traceLayout],
        }),
        compute: { module: trace, entryPoint: 'main' },
      });
      c.deposit = await device.createRenderPipelineAsync({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [c.depositLayout],
        }),
        vertex: { module: deposit, entryPoint: 'vertex' },
        fragment: {
          module: deposit,
          entryPoint: 'fragment',
          targets: [
            {
              format: 'rgba16float',
              blend: {
                color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              },
            },
          ],
        },
      });
      c.duckResolve = await ComputeKernel.create(
        device,
        'duck irradiance resolve',
        duckLightShader,
      );
      return c;
    } catch (error) {
      c.destroy();
      throw error;
    }
  }
  // Called when the renderer loads/replaces a mesh, before it caches surface bindings.
  setModel(triangles: GPUBuffer) {
    const vertices = Math.floor(triangles.size / 96) * 3;
    if (vertices !== this.duckVertices) {
      this.duckLighting.destroy();
      this.duckVertices = vertices;
      this.duckLighting = buffer(
        this.device,
        'duck surface irradiance',
        vertices * 16,
      );
    }
    this.clearCache();
  }
  encode(
    encoder: GPUCommandEncoder,
    volume: WebGPUVolume,
    sampler: GPUSampler,
    duck: GPUBuffer,
    bvh: GPUBuffer,
    triangles: GPUBuffer,
    count: number,
  ) {
    if (this.duckVertices !== Math.floor(triangles.size / 96) * 3)
      this.setModel(triangles);
    this.device.queue.writeBuffer(
      this.control,
      0,
      new Uint32Array([count, this.duckVertices, 0, 0]),
    );
    encoder.clearBuffer(this.duckDraw);
    let group = this.groups.get(duck);
    if (!group) {
      group = this.device.createBindGroup({
        layout: this.traceLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: volume.view },
          { binding: 2, resource: sampler },
          { binding: 3, resource: { buffer: volume.bounds } },
          { binding: 4, resource: { buffer: duck } },
          { binding: 5, resource: { buffer: bvh } },
          { binding: 6, resource: { buffer: triangles } },
          { binding: 12, resource: { buffer: this.photons } },
          { binding: 16, resource: { buffer: this.control } },
          { binding: 17, resource: { buffer: this.duckPhotons } },
          { binding: 18, resource: { buffer: this.duckDraw } },
        ],
      });
      this.groups.set(duck, group);
    }
    const trace = encoder.beginComputePass({ label: 'caustic light tracing' });
    trace.setPipeline(this.trace);
    trace.setBindGroup(0, group);
    trace.dispatchWorkgroups(
      Math.ceil(CAUSTIC_GRID[0] / 8),
      Math.ceil(CAUSTIC_GRID[1] / 8),
    );
    trace.end();
    // Direct rays and opaque shadows remain present even when the pool is empty.
    const pass = encoder.beginRenderPass({
      label: 'caustic floor map',
      colorAttachments: [
        {
          view: this.view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: [0, 0, 0, 0],
        },
      ],
    });
    pass.setPipeline(this.deposit);
    pass.setBindGroup(0, this.depositGroup);
    pass.draw((CAUSTIC_GRID[0] - 1) * (CAUSTIC_GRID[1] - 1) * 6);
    pass.end();
    const resolve = encoder.beginComputePass({
      label: 'duck irradiance resolve',
    });
    this.duckResolve.dispatch(
      resolve,
      {
        0: this.control,
        1: this.photons,
        2: this.duckPhotons,
        3: this.duckDraw,
        4: duck,
        5: triangles,
        6: this.duckLighting,
      },
      Math.ceil(this.duckVertices / 64),
    );
    resolve.end();
  }
  clearCache() {
    this.groups.clear();
    this.duckResolve?.clearCache();
  }
  destroy() {
    this.photons.destroy();
    this.control.destroy();
    this.texture.destroy();
    this.duckPhotons.destroy();
    this.duckDraw.destroy();
    this.duckLighting.destroy();
    this.clearCache();
  }
}
