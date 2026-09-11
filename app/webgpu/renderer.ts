import { buffer } from './compute.ts';
import { renderShader, particlesShader } from './render-shaders.ts';
import { detailShader } from './detail-shaders.ts';
import { WebGPUCaustics } from './caustics.ts';
import type { WebGPUSimulation } from './simulation.ts';
import type { WebGPUVolume } from './volume.ts';
export type Camera = {
  eye: number[];
  forward: number[];
  right: number[];
  up: number[];
};
export class WebGPURenderer {
  readonly uniform: GPUBuffer;
  private caustics!: WebGPUCaustics;
  private surface!: GPURenderPipeline;
  private selfOpticsSurface!: GPURenderPipeline;
  private particles!: GPURenderPipeline;
  private detailPipeline!: GPURenderPipeline;
  private detailLayout: GPUBindGroupLayout;
  private detailGroup: GPUBindGroup | null = null;
  private detailHits: GPUTexture | null = null;
  private detailNormals: GPUTexture | null = null;
  private detailDepth: GPUTexture | null = null;
  private surfaceLayout: GPUBindGroupLayout;
  private particleLayout: GPUBindGroupLayout;
  private densitySampler: GPUSampler;
  private albedoSampler: GPUSampler;
  private bvh: GPUBuffer;
  private triangles: GPUBuffer;
  private albedo: GPUTexture;
  private depth: GPUTexture | null = null;
  private width = 0;
  private height = 0;
  private surfaceGroups = new Map<GPUBuffer, GPUBindGroup>();
  private particleGroups = new Map<GPUBuffer, GPUBindGroup>();
  private constructor(readonlyDevice: GPUDevice, format: GPUTextureFormat) {
    this.device = readonlyDevice;
    this.format = format;
    const device = readonlyDevice;
    this.uniform = buffer(
      device,
      'scene uniforms',
      128,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.bvh = buffer(device, 'duck BVH', 16);
    this.triangles = buffer(device, 'duck triangles', 16);
    this.albedo = device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.writeTexture(
      { texture: this.albedo },
      new Uint8Array([255, 195, 20, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    );
    const filtered = device.features.has('float32-filterable');
    this.densitySampler = device.createSampler({
      minFilter: filtered ? 'linear' : 'nearest',
      magFilter: filtered ? 'linear' : 'nearest',
    });
    this.albedoSampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
    });
    const F = GPUShaderStage.FRAGMENT,
      V = GPUShaderStage.VERTEX;
    this.surfaceLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: F | V, buffer: { type: 'uniform' } },
        {
          binding: 1,
          visibility: F,
          texture: {
            viewDimension: '3d',
            sampleType: filtered ? 'float' : 'unfilterable-float',
          },
        },
        {
          binding: 2,
          visibility: F,
          sampler: { type: filtered ? 'filtering' : 'non-filtering' },
        },
        ...[3, 4, 5, 6].map((binding) => ({
          binding,
          visibility: F,
          buffer: { type: 'read-only-storage' as const },
        })),
        { binding: 7, visibility: F, texture: { sampleType: 'float' } },
        { binding: 8, visibility: F, sampler: { type: 'filtering' } },
        ...[9, 10].map((binding) => ({
          binding,
          visibility: F,
          texture: { sampleType: 'unfilterable-float' as const },
        })),
        { binding: 11, visibility: F, buffer: { type: 'read-only-storage' } },
        { binding: 12, visibility: F, texture: { sampleType: 'float' } },
        { binding: 13, visibility: F, sampler: { type: 'filtering' } },
        { binding: 14, visibility: F, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.particleLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: F | V, buffer: { type: 'uniform' } },
        { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.detailLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
        {
          binding: 1,
          visibility: V | F,
          buffer: { type: 'read-only-storage' },
        },
      ],
    });
  }
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;
  static async create(device: GPUDevice, format: GPUTextureFormat) {
    const r = new WebGPURenderer(device, format);
    try {
      r.caustics = await WebGPUCaustics.create(device, r.uniform);
      for (const [name, code, layout, compare] of [
        [
          'surface',
          renderShader(device.features.has('float32-filterable')),
          r.surfaceLayout,
          'always',
        ],
        [
          'selfOpticsSurface',
          renderShader(device.features.has('float32-filterable'), true),
          r.surfaceLayout,
          'always',
        ],
        ['particles', particlesShader, r.particleLayout, 'less'],
        ['detailPipeline', detailShader, r.detailLayout, 'less'],
      ] as const) {
        const shaderModule = device.createShaderModule({ label: name, code });
        const info = await shaderModule.getCompilationInfo();
        const errors = info.messages.filter((m) => m.type === 'error');
        if (errors.length)
          throw new Error(
            `${name}: ${errors.map((m) => `${m.lineNum} ${m.message}`).join('\n')}`,
          );
        r[name] = await device.createRenderPipelineAsync({
          label: name,
          layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
          vertex: { module: shaderModule, entryPoint: 'vertex' },
          fragment: {
            module: shaderModule,
            entryPoint: 'fragment',
            targets:
              name === 'detailPipeline'
                ? [{ format: 'rgba32float' }, { format: 'rgba16float' }]
                : [{ format }],
          },
          primitive: { topology: 'triangle-list' },
          depthStencil: {
            format: 'depth32float',
            depthWriteEnabled: true,
            depthCompare: compare,
          },
        });
      }
      return r;
    } catch (e) {
      r.destroy();
      throw e;
    }
  }
  async loadModel(signal?: AbortSignal) {
    const base = `${process.env.NEXT_PUBLIC_ASSET_BASE || ''}/models/duck/`;
    const get = async (path: string) => {
      const r = await fetch(base + path, { signal });
      if (!r.ok) throw new Error(`鸭子资源加载失败 (${r.status})`);
      return r;
    };
    const [bvh, triangles, image] = await Promise.all([
      get('bvh.bin').then((r) => r.arrayBuffer()),
      get('triangles.bin').then((r) => r.arrayBuffer()),
      get('DuckCM.png').then((r) => r.blob()),
    ]);
    const bitmap = await createImageBitmap(image, {
      imageOrientation: 'none',
      premultiplyAlpha: 'none',
      colorSpaceConversion: 'none',
    });
    if (signal?.aborted) {
      bitmap.close();
      throw new Error('Model loading cancelled');
    }
    this.setModel(bvh, triangles, bitmap);
    bitmap.close();
  }
  setModel(bvh: ArrayBuffer, triangles: ArrayBuffer, image?: ImageBitmap) {
    this.bvh.destroy();
    this.triangles.destroy();
    this.bvh = buffer(this.device, 'duck BVH', bvh.byteLength);
    this.triangles = buffer(
      this.device,
      'duck triangles',
      triangles.byteLength,
    );
    this.device.queue.writeBuffer(this.bvh, 0, bvh);
    this.device.queue.writeBuffer(this.triangles, 0, triangles);
    if (image) {
      this.albedo.destroy();
      this.albedo = this.device.createTexture({
        size: [image.width, image.height],
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.device.queue.copyExternalImageToTexture(
        { source: image },
        { texture: this.albedo },
        [image.width, image.height],
      );
    }
    this.surfaceGroups.clear();
    this.caustics.setModel(this.triangles);
    this.ready = true;
  }
  ready = false;
  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    sim: WebGPUSimulation,
    volume: WebGPUVolume,
    camera: Camera,
    width: number,
    height: number,
    options: {
      light: number;
      reflection: boolean;
      selfReflection?: boolean;
      selfShadow?: boolean;
      caustics: boolean;
      particles: boolean;
      brush?: number[];
    },
  ) {
    if (width !== this.width || height !== this.height) {
      this.depth?.destroy();
      this.depth = this.device.createTexture({
        label: 'scene depth',
        size: [width, height],
        format: 'depth32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.width = width;
      this.height = height;
      this.detailHits?.destroy();
      this.detailNormals?.destroy();
      this.detailDepth?.destroy();
      this.detailHits = this.device.createTexture({
        label: 'detail hit and coverage',
        size: [width, height],
        format: 'rgba32float',
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      });
      this.detailNormals = this.device.createTexture({
        label: 'detail normal and chord',
        size: [width, height],
        format: 'rgba16float',
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      });
      this.detailDepth = this.device.createTexture({
        size: [width, height],
        format: 'depth32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.surfaceGroups.clear();
    }
    const u = new Float32Array(32);
    u.set([...camera.eye, +volume.detailsEnabled], 0);
    u.set([...camera.forward, +(options.selfReflection ?? false)], 4);
    u.set([...camera.right, +(options.selfShadow ?? false)], 8);
    u.set([...camera.up, 0], 12);
    u.set([width, height, width / height > 1.18 ? 0.19 : 0, sim.time], 16);
    u.set(
      [
        options.light,
        +options.reflection,
        +options.caustics,
        +options.particles,
      ],
      20,
    );
    u.set(options.brush ?? [0, -0.25, 0, 0], 24);
    u.set([1.15, +this.ready, 0.021, Math.cbrt(10000 / sim.quality)], 28);
    this.device.queue.writeBuffer(this.uniform, 0, u);
    if (!options.particles && options.light > 0)
      this.caustics.encode(
        encoder,
        volume,
        this.densitySampler,
        sim.duck,
        this.bvh,
        this.triangles,
        sim.count,
      );
    if (volume.detailsEnabled && !options.particles) {
      this.detailGroup ??= this.device.createBindGroup({
        layout: this.detailLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: volume.details } },
        ],
      });
      const detailPass = encoder.beginRenderPass({
        label: 'primary droplet silhouettes',
        colorAttachments: [this.detailHits!, this.detailNormals!].map(
          (texture) => ({
            view: texture.createView(),
            loadOp: 'clear' as const,
            storeOp: 'store' as const,
            clearValue: [0, 0, 0, 0],
          }),
        ),
        depthStencilAttachment: {
          view: this.detailDepth!.createView(),
          depthLoadOp: 'clear',
          depthStoreOp: 'discard',
          depthClearValue: 1,
        },
      });
      detailPass.setPipeline(this.detailPipeline);
      detailPass.setBindGroup(0, this.detailGroup);
      detailPass.drawIndirect(volume.detailDraw, 0);
      detailPass.end();
    }
    let surface = this.surfaceGroups.get(sim.duck);
    if (!surface) {
      surface = this.device.createBindGroup({
        layout: this.surfaceLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: volume.view },
          { binding: 2, resource: this.densitySampler },
          { binding: 3, resource: { buffer: volume.bounds } },
          { binding: 4, resource: { buffer: sim.duck } },
          { binding: 5, resource: { buffer: this.bvh } },
          { binding: 6, resource: { buffer: this.triangles } },
          { binding: 7, resource: this.albedo.createView() },
          { binding: 8, resource: this.albedoSampler },
          { binding: 9, resource: this.detailHits!.createView() },
          { binding: 10, resource: this.detailNormals!.createView() },
          { binding: 11, resource: { buffer: volume.details } },
          { binding: 12, resource: this.caustics.view },
          { binding: 13, resource: this.albedoSampler },
          { binding: 14, resource: { buffer: this.caustics.duckLighting } },
        ],
      });
      this.surfaceGroups.set(sim.duck, surface);
    }
    const pass = encoder.beginRenderPass({
      label: 'water and duck',
      colorAttachments: [
        {
          view: target,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: [0.02, 0.03, 0.04, 1],
        },
      ],
      depthStencilAttachment: {
        view: this.depth!.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1,
      },
    });
    pass.setPipeline(
      !options.particles &&
        (options.selfShadow || (options.reflection && options.selfReflection))
        ? this.selfOpticsSurface
        : this.surface,
    );
    pass.setBindGroup(0, surface);
    pass.draw(3);
    if (options.particles) {
      let group = this.particleGroups.get(sim.state);
      if (!group) {
        group = this.device.createBindGroup({
          layout: this.particleLayout,
          entries: [
            { binding: 0, resource: { buffer: this.uniform } },
            { binding: 1, resource: { buffer: sim.state } },
          ],
        });
        this.particleGroups.set(sim.state, group);
      }
      pass.setPipeline(this.particles);
      pass.setBindGroup(0, group);
      pass.draw(6, sim.count);
    }
    pass.end();
  }
  destroy() {
    this.caustics?.destroy();
    this.uniform.destroy();
    this.bvh.destroy();
    this.triangles.destroy();
    this.albedo.destroy();
    this.depth?.destroy();
    this.detailHits?.destroy();
    this.detailNormals?.destroy();
    this.detailDepth?.destroy();
    this.detailGroup = null;
    this.surfaceGroups.clear();
    this.particleGroups.clear();
  }
}
