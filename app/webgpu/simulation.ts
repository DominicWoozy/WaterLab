import {
  CAPACITY,
  GRID_CELLS,
  GRID_STORAGE_WORDS,
  WORKGROUP,
} from './common.ts';
import { ComputeKernel, buffer } from './compute.ts';
import { gridShaders } from './grid-shaders.ts';
import { fluidShaders } from './fluid-shaders.ts';
import { duckShaders } from './duck-shaders.ts';
import { capillaryShaders } from './capillary-shaders.ts';
import type { FluidJob } from '../fluid-runtime.ts';
export type WebGPUQuality = 15000 | 30000 | 50000;
export const PHYSICS_SUBSTEPS = 3;
export const PHYSICS_DT = 1 / (60 * PHYSICS_SUBSTEPS);
// Two scheduled ticks can share a submission: each substep has regular and
// reaction-reset uniforms. Rendering and reset must not overwrite those slots.
export const RENDER_PARAMETER_SLOT = 4 * PHYSICS_SUBSTEPS;
const RESET_PARAMETER_SLOT = RENDER_PARAMETER_SLOT + 1;
export type StepInput = Pick<FluidJob, 'forces' | 'brush'> & {
  shake?: number;
  splash?: number[];
  pourAt?: number[];
  previousCount?: number;
  surfaceTension?: boolean;
  capillaryMode?: 'implicit' | 'explicit';
};
export class WebGPUSimulation {
  count: number = 50000;
  quality: WebGPUQuality = 50000;
  time = 0;
  gravity = 9.8;
  state: GPUBuffer;
  spare: GPUBuffer;
  duck: GPUBuffer;
  duckSpare: GPUBuffer;
  readonly starts: GPUBuffer;
  readonly parameters: GPUBuffer[];
  private counts: GPUBuffer;
  private cursor: GPUBuffer;
  private totals: GPUBuffer;
  private lambda: GPUBuffer;
  private factor: GPUBuffer;
  readonly surface: GPUBuffer;
  readonly capillaryRhs: GPUBuffer;
  capillaryGuess: GPUBuffer;
  private capillaryNext: GPUBuffer;
  private reactions: GPUBuffer;
  private reactionSpare: GPUBuffer;
  private reactionGroups: GPUBuffer;
  private reactionTotal: GPUBuffer;
  private owned: GPUBuffer[] = [];
  private kernels = new Map<string, ComputeKernel>();
  readonly device: GPUDevice;
  private constructor(device: GPUDevice) {
    this.device = device;
    const alloc = (name: string, size: number, usage?: number) => {
      const b = buffer(device, name, size, usage);
      this.owned.push(b);
      return b;
    };
    this.state = alloc('particles-a', CAPACITY * 48);
    this.spare = alloc('particles-b', CAPACITY * 48);
    this.duck = alloc('duck-a', 64);
    this.duckSpare = alloc('duck-b', 64);
    this.starts = alloc(
      'cell starts and neighbor cache',
      GRID_STORAGE_WORDS * 4,
    );
    this.counts = alloc('cell counts', GRID_CELLS * 4);
    this.cursor = alloc('cell cursors', GRID_CELLS * 4);
    this.totals = alloc('block totals', 128 * 4);
    this.lambda = alloc('pressure', CAPACITY * 16);
    this.factor = alloc('divergence factors', CAPACITY * 16);
    this.surface = alloc('surface normals and density', CAPACITY * 16);
    this.capillaryRhs = alloc(
      'capillary rhs and inverse diagonal',
      CAPACITY * 16,
    );
    this.capillaryGuess = alloc('capillary velocity a', CAPACITY * 16);
    this.capillaryNext = alloc('capillary velocity b', CAPACITY * 16);
    this.reactions = alloc('particle reactions', CAPACITY * 32);
    this.reactionSpare = alloc('sorted reactions', CAPACITY * 32);
    this.reactionGroups = alloc('reaction groups', 256 * 32);
    this.reactionTotal = alloc('reaction sum', 32);
    this.parameters = Array.from({ length: RESET_PARAMETER_SLOT + 1 }, (_, i) =>
      alloc(
        `parameters-${i}`,
        96,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      ),
    );
  }
  static async create(device: GPUDevice) {
    const sim = new WebGPUSimulation(device);
    try {
      for (const [name, code] of Object.entries({
        ...gridShaders,
        ...fluidShaders,
        ...duckShaders,
        ...capillaryShaders,
      }))
        sim.kernels.set(name, await ComputeKernel.create(device, name, code));
      return sim;
    } catch (e) {
      sim.destroy();
      throw e;
    }
  }
  writeParameters(slot: number, input: StepInput, resetReaction = false) {
    const data = new ArrayBuffer(96),
      u = new Uint32Array(data),
      f = new Float32Array(data);
    u.set([
      this.count,
      this.quality,
      input.previousCount ?? this.count,
      +resetReaction,
    ]);
    f.set(
      [
        PHYSICS_DT,
        this.time,
        Math.cbrt(10000 / this.quality),
        input.forces.gravity,
      ],
      4,
    );
    f.set(
      [
        input.forces.viscosity,
        input.forces.agitation,
        input.shake ?? 0,
        (input.surfaceTension ?? true)
          ? input.capillaryMode === 'implicit'
            ? 2
            : 1
          : 0,
      ],
      8,
    );
    const b = input.brush;
    f.set(b?.mode === 'stir' ? [b.x, b.y, b.z, b.strength] : [0, 0, 0, 0], 12);
    f.set([b?.dx ?? 0, b?.dz ?? 0, ...(input.pourAt ?? [-0.7, 0])], 16);
    f.set([...(input.splash ?? [0, 0, 0]), 0], 20);
    this.device.queue.writeBuffer(this.parameters[slot], 0, data);
    return this.parameters[slot];
  }
  private run(
    pass: GPUComputePassEncoder,
    name: string,
    p: GPUBuffer,
    resources: Record<number, GPUBuffer> = {},
    groups = Math.ceil(this.count / WORKGROUP),
  ) {
    const defaults = {
      0: p,
      1: this.state,
      2: this.spare,
      3: this.starts,
      4: this.factor,
      5: this.lambda,
      6: this.duck,
      7: this.reactions,
      8: this.surface,
    };
    this.kernels
      .get(name)!
      .dispatch(pass, { ...defaults, ...resources }, groups);
  }
  private swap() {
    [this.state, this.spare] = [this.spare, this.state];
  }
  private swapDuck() {
    [this.duck, this.duckSpare] = [this.duckSpare, this.duck];
  }
  solveCapillary(pass: GPUComputePassEncoder, p: GPUBuffer) {
    const bindings = () => ({
      4: this.surface,
      8: this.factor,
      5: this.capillaryRhs,
      6: this.capillaryGuess,
      7: this.capillaryNext,
    });
    const swap = () => {
      [this.capillaryGuess, this.capillaryNext] = [
        this.capillaryNext,
        this.capillaryGuess,
      ];
    };
    this.run(pass, 'capillaryPrepare', p, bindings());
    swap();
    // Optional bounded comparison path; the interactive default fuses explicit
    // cohesion into viscosity. No global reduction or CPU synchronization.
    for (let iteration = 0; iteration < 4; iteration++) {
      this.run(pass, 'capillaryIterate', p, bindings());
      swap();
    }
    this.run(pass, 'capillaryApply', p, bindings());
    this.swap();
  }
  reset(encoder: GPUCommandEncoder, quality: WebGPUQuality = 50000) {
    this.quality = quality;
    this.count = quality;
    this.time = 0;
    const p = this.writeParameters(RESET_PARAMETER_SLOT, {
      forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 },
    });
    const pass = encoder.beginComputePass({ label: 'reset' });
    this.run(pass, 'initialize', p);
    this.swap();
    this.run(
      pass,
      'duckInitialize',
      p,
      { 1: this.duck, 2: this.duckSpare, 3: this.reactionTotal },
      1,
    );
    this.swapDuck();
    pass.end();
  }
  buildGrid(
    encoder: GPUCommandEncoder,
    p = this.parameters[RENDER_PARAMETER_SLOT],
    reorderReactions = false,
  ) {
    encoder.clearBuffer(this.counts);
    encoder.clearBuffer(this.cursor);
    const pass = encoder.beginComputePass({ label: 'count-scan-scatter' });
    this.run(pass, 'count', p, { 2: this.counts });
    this.run(
      pass,
      'scan',
      p,
      { 1: this.counts, 2: this.starts, 3: this.totals },
      120,
    );
    this.run(pass, 'scanTotals', p, { 1: this.totals }, 1);
    this.run(pass, 'add', p, { 1: this.starts, 2: this.totals }, 120);
    if (reorderReactions) {
      this.run(pass, 'scatterReactions', p, {
        3: this.starts,
        4: this.cursor,
        5: this.reactions,
        6: this.reactionSpare,
      });
      [this.reactions, this.reactionSpare] = [
        this.reactionSpare,
        this.reactions,
      ];
    } else this.run(pass, 'scatter', p, { 3: this.starts, 4: this.cursor });
    this.swap();
    pass.end();
  }
  step(encoder: GPUCommandEncoder, input: StepInput, slot = 0) {
    // A splash is an impulse, while brush/shake/gravity are continuous forces.
    // Newly poured particles must be initialized only in the first substep.
    for (let substep = 0; substep < PHYSICS_SUBSTEPS; substep++)
      this.substep(
        encoder,
        substep === 0
          ? input
          : { ...input, splash: undefined, previousCount: this.count },
        slot * PHYSICS_SUBSTEPS + substep,
      );
  }
  private substep(encoder: GPUCommandEncoder, input: StepInput, slot: number) {
    this.gravity = input.forces.gravity;
    const p = this.writeParameters(slot * 2, input),
      first = this.writeParameters(slot * 2 + 1, input, true);
    let pass = encoder.beginComputePass({ label: 'predict' });
    this.run(
      pass,
      'duckPredict',
      p,
      { 1: this.duck, 2: this.duckSpare, 3: this.reactionTotal },
      1,
    );
    this.swapDuck();
    this.run(pass, 'predict', p);
    this.swap();
    pass.end();
    this.buildGrid(encoder, p);
    pass = encoder.beginComputePass({ label: 'fluid-pressure-and-duck' });
    // Three 1/180-second steps with four relaxed rounds each keep compression
    // below the equilibrium target with fewer traversals than two deep solves.
    for (let iteration = 0; iteration < 4; iteration++) {
      this.run(pass, 'lambda', p);
      this.run(pass, 'correct', iteration === 0 ? first : p, {
        4: this.lambda,
        5: this.factor,
      });
      this.swap();
      // One correction can move a particle by .042*scale including duck contact.
      // Refresh before this exceeds the .055*scale cell support margin.
      if (iteration % 2 === 1) {
        pass.end();
        this.buildGrid(encoder, p, true);
        pass = encoder.beginComputePass({
          label: 'pressure-final-and-divergence',
        });
      }
    }
    // Positions remain fixed: gather exact neighbors and factors while
    // reconstructing velocity, then reuse the list through all projections.
    this.run(pass, 'prepareVelocity', p, { 5: this.factor });
    this.swap();
    this.run(pass, 'viscosity', p);
    this.swap();
    if ((input.surfaceTension ?? true) && input.capillaryMode === 'implicit')
      this.solveCapillary(pass, p);
    for (let iteration = 0; iteration < 2; iteration++) {
      this.run(pass, 'residual', p);
      this.run(pass, 'project', p, { 4: this.lambda, 5: this.factor });
      this.swap();
    }
    this.run(
      pass,
      'reduceParticles',
      p,
      { 1: this.reactions, 2: this.reactionGroups },
      Math.ceil(this.count / 256),
    );
    this.run(
      pass,
      'reduceGroups',
      p,
      { 1: this.reactionGroups, 2: this.reactionTotal },
      1,
    );
    this.run(
      pass,
      'duckIntegrate',
      p,
      { 1: this.duck, 2: this.duckSpare, 3: this.reactionTotal },
      1,
    );
    this.swapDuck();
    pass.end();
    this.time += PHYSICS_DT;
  }
  destroy() {
    this.owned.forEach((b) => b.destroy());
    this.kernels.forEach((k) => k.clearCache());
  }
}
