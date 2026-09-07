import {
  ParticleFluid,
  DEFAULT_COUNT,
  type FluidForces,
} from './fluid-simulation.ts';
import { FluidVolume, VOLUME_SIZE } from './fluid-volume.ts';
export type FluidAction = {
  type: 'reset' | 'shake' | 'pour' | 'drain' | 'splash';
  x?: number;
  z?: number;
  strength?: number;
  amount?: number;
};
export type FluidJob = {
  elapsed: number;
  speed: number;
  paused: boolean;
  forces: FluidForces;
  particles: boolean;
  brush?: {
    x: number;
    y: number;
    z: number;
    dx: number;
    dz: number;
    strength: number;
    mode: 'stir' | 'pour';
  };
  actions: FluidAction[];
  recycleVolume?: ArrayBuffer;
  recyclePositions?: ArrayBuffer;
};
export type FluidFrame = {
  time: number;
  count: number;
  top: number;
  computeMs: number;
  volume: ArrayBuffer | null;
  positions: ArrayBuffer | null;
};
export class FluidRuntime {
  readonly fluid = new ParticleFluid(DEFAULT_COUNT);
  private volume = new FluidVolume();
  private accumulator = 0;
  private pendingPour = 0;
  private pourX = -0.7;
  private pourZ = 0;
  private lastVolumeTime = -1;
  private lastVolumeCount = -1;
  private availableVolume: ArrayBuffer | null = null;
  private availablePositions: ArrayBuffer | null = null;
  run(job: FluidJob): FluidFrame {
    const started = performance.now();
    if (job.recycleVolume) this.availableVolume = job.recycleVolume;
    if (job.recyclePositions) this.availablePositions = job.recyclePositions;
    for (const action of job.actions) {
      if (action.type === 'reset') {
        this.fluid.reset();
        this.accumulator = 0;
        this.pendingPour = 0;
        this.lastVolumeTime = -1;
      }
      if (action.type === 'shake') this.fluid.shake();
      if (action.type === 'splash')
        this.fluid.splash(action.x ?? 0, action.z ?? 0, action.strength ?? 1);
      if (action.type === 'pour') {
        this.pendingPour = Math.min(
          2000,
          this.pendingPour + (action.amount ?? 500),
        );
        this.pourX = action.x ?? -0.7;
        this.pourZ = action.z ?? 0;
      }
      if (action.type === 'drain') this.fluid.drain(action.amount ?? 500);
    }
    if (!job.paused) {
      this.accumulator = Math.min(
        0.05,
        this.accumulator + Math.max(0, Math.min(0.1, job.elapsed)) * job.speed,
      );
      while (this.accumulator + 1e-8 >= 1 / 60) {
        const b = job.brush;
        if (b) {
          if (b.mode === 'pour') this.fluid.pour(b.x, b.z, 14);
          else this.fluid.stir(b.x, b.y, b.z, b.dx, b.dz, b.strength, 1 / 60);
        }
        if (this.pendingPour > 0) {
          const n = Math.min(18, this.pendingPour);
          this.fluid.pour(this.pourX, this.pourZ, n);
          this.pendingPour -= n;
        }
        this.fluid.step(1 / 60, job.forces);
        this.accumulator -= 1 / 60;
      }
    } else this.accumulator = 0;
    let volume: ArrayBuffer | null = null,
      positions: ArrayBuffer | null = null;
    if (
      !job.particles &&
      (this.lastVolumeTime !== this.fluid.time ||
        this.lastVolumeCount !== this.fluid.count)
    ) {
      const length =
        this.volume.data.length ||
        VOLUME_SIZE[0] * VOLUME_SIZE[1] * VOLUME_SIZE[2];
      this.volume.data = this.availableVolume
        ? new Float32Array(this.availableVolume)
        : new Float32Array(length);
      this.availableVolume = null;
      this.volume.rebuild(
        this.fluid.positions,
        this.fluid.count,
        this.fluid.densities,
      );
      volume = this.volume.data.buffer as ArrayBuffer;
      this.lastVolumeTime = this.fluid.time;
      this.lastVolumeCount = this.fluid.count;
    }
    if (job.particles) {
      const length = this.fluid.positions.byteLength;
      positions =
        this.availablePositions?.byteLength === length
          ? this.availablePositions
          : new ArrayBuffer(length);
      this.availablePositions = null;
      new Float32Array(positions).set(this.fluid.positions);
    }
    return {
      time: this.fluid.time,
      count: this.fluid.count,
      top: this.volume.top,
      computeMs: performance.now() - started,
      volume,
      positions,
    };
  }
}
