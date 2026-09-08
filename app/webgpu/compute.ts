/// <reference types="@webgpu/types" />
export class ComputeKernel {
  private cache = new Map<string, GPUBindGroup>();
  private static ids = new WeakMap<object, number>();
  private static next = 1;
  readonly device: GPUDevice;
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
  readonly bindings: number[];
  private constructor(
    device: GPUDevice,
    pipeline: GPUComputePipeline,
    layout: GPUBindGroupLayout,
    bindings: number[],
  ) {
    this.device = device;
    this.pipeline = pipeline;
    this.layout = layout;
    this.bindings = bindings;
  }
  static async create(device: GPUDevice, name: string, code: string) {
    const entries: GPUBindGroupLayoutEntry[] = [];
    for (const match of code.matchAll(
      /@group\(0\)\s*@binding\((\d+)\)\s*var<(uniform|storage)(?:,\s*(read|read_write))?>/g,
    )) {
      entries.push({
        binding: Number(match[1]),
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type:
            match[2] === 'uniform'
              ? 'uniform'
              : match[3] === 'read_write'
                ? 'storage'
                : 'read-only-storage',
        },
      });
    }
    const layout = device.createBindGroupLayout({ label: name, entries });
    const shaderModule = device.createShaderModule({ label: name, code });
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length)
      throw new Error(
        `${name}: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n')}`,
      );
    const pipeline = await device.createComputePipelineAsync({
      label: name,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: shaderModule, entryPoint: 'main' },
    });
    return new ComputeKernel(
      device,
      pipeline,
      layout,
      entries.map((e) => e.binding),
    );
  }
  dispatch(
    pass: GPUComputePassEncoder,
    resources: Record<number, GPUBuffer>,
    x: number,
    y = 1,
    z = 1,
  ) {
    const key = this.bindings
      .map((binding) => {
        const b = resources[binding];
        if (!b) throw new Error(`Missing compute binding ${binding}`);
        if (!ComputeKernel.ids.has(b))
          ComputeKernel.ids.set(b, ComputeKernel.next++);
        return ComputeKernel.ids.get(b);
      })
      .join(',');
    let group = this.cache.get(key);
    if (!group) {
      group = this.device.createBindGroup({
        layout: this.layout,
        entries: this.bindings.map((binding) => ({
          binding,
          resource: { buffer: resources[binding] },
        })),
      });
      this.cache.set(key, group);
    }
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.max(1, x), y, z);
  }
  clearCache() {
    this.cache.clear();
  }
}
export function buffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage = GPUBufferUsage.STORAGE |
    GPUBufferUsage.COPY_SRC |
    GPUBufferUsage.COPY_DST,
) {
  return device.createBuffer({
    label,
    size: Math.max(16, Math.ceil(size / 16) * 16),
    usage,
  });
}
