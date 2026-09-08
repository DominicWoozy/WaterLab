import { buffer, ComputeKernel } from './compute.ts';
import { CAPACITY } from './common.ts';
import { volumeShaders } from './volume-shaders.ts';
import type { WebGPUSimulation } from './simulation.ts';
export class WebGPUVolume {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly shapes: GPUBuffer;
  readonly bounds: GPUBuffer;
  readonly density: GPUBuffer;
  private temp: GPUBuffer;
  private filtered: GPUBuffer;
  private kernels = new Map<string, ComputeKernel>();
  private constructor(device: GPUDevice) {
    this.shapes = buffer(
      device,
      'particle reconstruction shapes',
      CAPACITY * 64,
    );
    this.bounds = buffer(device, 'water height', 16);
    this.density = buffer(device, 'density field', 128 * 160 * 96 * 4);
    this.temp = buffer(device, 'density filter xz', 128 * 160 * 96 * 4);
    this.filtered = buffer(device, 'density filter y', 128 * 160 * 96 * 4);
    this.texture = device.createTexture({
      label: 'water density',
      size: [128, 160, 96],
      dimension: '3d',
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.view = this.texture.createView();
  }
  static async create(device: GPUDevice) {
    const volume = new WebGPUVolume(device);
    try {
      for (const [name, code] of Object.entries(volumeShaders))
        volume.kernels.set(
          name,
          await ComputeKernel.create(device, name, code),
        );
      return volume;
    } catch (e) {
      volume.destroy();
      throw e;
    }
  }
  encode(encoder: GPUCommandEncoder, sim: WebGPUSimulation) {
    const p = sim.writeParameters(4, {
      forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 },
    });
    // This fresh grid makes render-kernel gathering exact after pressure/contact corrections.
    sim.buildGrid(encoder, p);
    encoder.clearBuffer(this.bounds);
    const pass = encoder.beginComputePass({
      label: 'anisotropy-density-filter',
    });
    this.kernels
      .get('geometry')!
      .dispatch(
        pass,
        { 0: p, 1: sim.state, 2: this.shapes, 3: sim.starts, 4: this.bounds },
        Math.ceil(sim.count / 128),
      );
    this.kernels.get('density')!.dispatch(
      pass,
      {
        0: p,
        1: this.shapes,
        2: sim.starts,
        3: this.density,
        4: this.bounds,
      },
      16,
      40,
      96,
    );
    const inputs = [this.density, this.temp, this.filtered],
      outputs = [this.temp, this.filtered, this.temp];
    for (let axis = 0; axis < 3; axis++)
      this.kernels
        .get(`filter${axis}`)!
        .dispatch(
          pass,
          { 0: p, 1: inputs[axis], 2: this.density, 3: outputs[axis] },
          16,
          40,
          24,
        );
    pass.end();
    encoder.copyBufferToTexture(
      { buffer: this.temp, bytesPerRow: 512, rowsPerImage: 160 },
      { texture: this.texture },
      [128, 160, 96],
    );
  }
  destroy() {
    [this.shapes, this.bounds, this.density, this.temp, this.filtered].forEach(
      (b) => b.destroy(),
    );
    this.texture.destroy();
    this.kernels.forEach((k) => k.clearCache());
  }
}
