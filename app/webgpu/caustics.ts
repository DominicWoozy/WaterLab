import { buffer } from './compute.ts';
import {
  CAUSTIC_GRID,
  CAUSTIC_MAP,
  causticTraceShader,
  causticDepositShader,
} from './caustic-shaders.ts';
import type { WebGPUVolume } from './volume.ts';

export class WebGPUCaustics {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly photons: GPUBuffer;
  private trace!: GPUComputePipeline;
  private deposit!: GPURenderPipeline;
  private traceLayout: GPUBindGroupLayout;
  private depositLayout: GPUBindGroupLayout;
  private depositGroup: GPUBindGroup;
  private groups = new Map<GPUBuffer, GPUBindGroup>();
  private device: GPUDevice;
  private uniform: GPUBuffer;

  private constructor(device: GPUDevice, uniform: GPUBuffer) {
    this.device = device;
    this.uniform = uniform;
    const filtered = device.features.has('float32-filterable');
    this.photons = buffer(
      device,
      'refracted photon wavefront',
      CAUSTIC_GRID[0] * CAUSTIC_GRID[1] * 48,
    );
    this.texture = device.createTexture({
      label: 'pool floor caustic irradiance',
      size: [...CAUSTIC_MAP],
      format: 'rgba16float',
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    this.view = this.texture.createView();
    const C = GPUShaderStage.COMPUTE;
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
        { binding: 12, visibility: C, buffer: { type: 'storage' } },
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
        const info = await shader.getCompilationInfo();
        const errors = info.messages.filter((m) => m.type === 'error');
        if (errors.length)
          throw new Error(
            errors.map((m) => `${m.lineNum}: ${m.message}`).join('\n'),
          );
      }
      c.trace = await device.createComputePipelineAsync({
        label: 'caustic surface rays',
        layout: device.createPipelineLayout({
          bindGroupLayouts: [c.traceLayout],
        }),
        compute: { module: trace, entryPoint: 'main' },
      });
      c.deposit = await device.createRenderPipelineAsync({
        label: 'caustic wavefront deposition',
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
      return c;
    } catch (error) {
      c.destroy();
      throw error;
    }
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
    if (count > 0) {
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
          ],
        });
        this.groups.set(duck, group);
      }
      const pass = encoder.beginComputePass({ label: 'caustic light tracing' });
      pass.setPipeline(this.trace);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(
        Math.ceil(CAUSTIC_GRID[0] / 8),
        Math.ceil(CAUSTIC_GRID[1] / 8),
      );
      pass.end();
    }
    // Clear even an empty pool, so drained/reset water cannot retain old light.
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
    if (count > 0) {
      pass.setPipeline(this.deposit);
      pass.setBindGroup(0, this.depositGroup);
      pass.draw((CAUSTIC_GRID[0] - 1) * (CAUSTIC_GRID[1] - 1) * 6);
    }
    pass.end();
  }
  clearCache() {
    this.groups.clear();
  }
  destroy() {
    this.photons.destroy();
    this.texture.destroy();
    this.groups.clear();
  }
}
