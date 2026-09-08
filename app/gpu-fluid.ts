import {
  GPU_ATLAS_SIZE,
  GPU_SLICES_PER_PARTICLE,
} from './gpu-volume-config.ts';
import * as shaders from './gpu-fluid-shaders.ts';
import type { FluidAction, FluidJob } from './fluid-runtime.ts';
import {
  GPU_CAPACITY as CAPACITY,
  GPU_DEFAULT_COUNT as DEFAULT_COUNT,
  PARTICLE_WIDTH,
  PARTICLE_HEIGHT,
  type ParticleQuality,
} from './gpu-particle-config.ts';
export type GpuFluidAction =
  | FluidAction
  | { type: 'quality'; count: ParticleQuality };
type Target = {
  texture: WebGLTexture;
  framebuffer: WebGLFramebuffer;
  width: number;
  height: number;
};
/** JS submits fixed GPU passes and small input uniforms; it never solves particles or reads state. */
export class GpuFluid {
  count = DEFAULT_COUNT;
  time = 0;
  quality: ParticleQuality = DEFAULT_COUNT;
  private sortCount = 16384;
  private gl: WebGL2RenderingContext;
  private targets: Target[] = [];
  private programs = new Map<string, WebGLProgram>();
  private scalarValues = new Map<WebGLProgram, Map<string, number>>();
  private activeProgram: WebGLProgram | null = null;
  private viewportWidth = 0;
  private viewportHeight = 0;
  private computeActive = false;
  private uniforms = new Map<
    WebGLProgram,
    Map<string, WebGLUniformLocation | null>
  >();
  private vao: WebGLVertexArrayObject;
  private position: Target;
  private predicted: Target;
  private correction: Target;
  private velocity: Target;
  private velocityTemp: Target;
  private keys: Target;
  private keysTemp: Target;
  private ranges: Target;
  private lambda: Target;
  private divergenceFactor: Target;
  private atlas: Target;
  private surfaceTemp: Target;
  private surfaceFiltered: Target;
  private bounds: Target[] = [];
  private duckState: Target;
  private duckTemp: Target;
  private reactions: Target[][] = [];
  private reduction: Target[][] = [];
  private geometry: Target[] = [];
  private sortedPosition: Target;
  private sortedOld: Target;
  private sortedVelocity: Target;
  private accumulator = 0;
  private pendingPour = 0;
  private pourAt = [-0.7, 0];
  private shakeUntil = 0;
  private splash = [0, 0, 0];
  private dirty = true;
  private densityDirty = true;
  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    if (!gl.getExtension('EXT_color_buffer_float'))
      throw new Error(
        'GPU 流体需要浮点渲染支持，请开启浏览器硬件加速或使用支持该功能的设备。',
      );
    this.vao = gl.createVertexArray()!;
    try {
      this.duckState = this.target(4, 1);
      this.duckTemp = this.target(4, 1);
      this.reactions = [
        [this.target(), this.target()],
        [this.target(), this.target()],
      ];
      for (
        let width = 128, height = 64;
        width >= 1;
        width /= 2, height = Math.max(1, height / 2)
      )
        this.reduction.push([
          this.target(width, height),
          this.target(width, height),
        ]);
      this.position = this.target();
      this.predicted = this.target();
      this.sortedPosition = this.target();
      this.sortedOld = this.target();
      this.sortedVelocity = this.target();
      this.correction = this.target();
      this.velocity = this.target();
      this.velocityTemp = this.target();
      this.keys = this.target();
      this.keysTemp = this.target();
      this.ranges = this.target(PARTICLE_WIDTH, 120);
      this.lambda = this.target();
      this.divergenceFactor = this.target();
      this.atlas = this.target(...GPU_ATLAS_SIZE, true);
      this.surfaceTemp = this.target(...GPU_ATLAS_SIZE, true);
      this.surfaceFiltered = this.target(...GPU_ATLAS_SIZE, true);
      for (
        let width = 128, height = 64;
        width >= 1;
        width /= 2, height = Math.max(1, height / 2)
      )
        this.bounds.push(this.target(width, height));
      for (let i = 0; i < 4; i++) this.geometry.push(this.target());
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.geometry[0].framebuffer);
      for (let i = 1; i < 4; i++)
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0 + i,
          gl.TEXTURE_2D,
          this.geometry[i].texture,
          0,
        );
      gl.drawBuffers([0, 1, 2, 3].map((i) => gl.COLOR_ATTACHMENT0 + i));
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('当前设备无法创建水面重建缓冲。');
      for (const name of [
        'duckInitialize',
        'duckPredict',
        'duckReduce',
        'duckIntegrate',
        'reorder',
        'geometry',
        'bounds',
        'initialize',
        'predict',
        'key',
        'sort',
        'sortMerge',
        'ranges',
        'lambda',
        'correct',
        'velocity',
        'viscosity',
        'divergenceFactor',
        'divergenceResidual',
        'divergenceProject',
        'volume',
        'surfaceFilter',
      ]) {
        const sources = shaders as unknown as Record<string, string>;
        this.programs.set(
          name,
          this.compile(
            name === 'volume' ? shaders.volumeVertex : shaders.computeVertex,
            sources[name + 'Fragment'],
          ),
        );
      }
      this.reset();
    } catch (error) {
      this.destroy();
      throw error;
    }
  }
  get duck() {
    return this.duckState.texture;
  }
  get positions() {
    return this.position.texture;
  }
  get volume() {
    return this.surfaceTemp.texture;
  }
  get volumeBounds() {
    return this.bounds[this.bounds.length - 1].texture;
  }
  private target(
    width = PARTICLE_WIDTH,
    height = PARTICLE_HEIGHT,
    half = false,
  ): Target {
    const gl = this.gl,
      texture = gl.createTexture()!,
      framebuffer = gl.createFramebuffer()!;
    const target = { texture, framebuffer, width, height };
    this.targets.push(target);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(
      gl.TEXTURE_2D,
      1,
      half ? gl.R16F : gl.RGBA32F,
      width,
      height,
    );
    gl.texParameteri(
      gl.TEXTURE_2D,
      gl.TEXTURE_MIN_FILTER,
      half ? gl.LINEAR : gl.NEAREST,
    );
    gl.texParameteri(
      gl.TEXTURE_2D,
      gl.TEXTURE_MAG_FILTER,
      half ? gl.LINEAR : gl.NEAREST,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0,
    );
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
      throw new Error('当前设备无法创建 GPU 流体浮点缓冲。');
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return target;
  }
  private compile(vertex: string, fragment: string) {
    const gl = this.gl,
      program = gl.createProgram()!;
    try {
      for (const [type, source] of [
        [gl.VERTEX_SHADER, vertex],
        [gl.FRAGMENT_SHADER, fragment],
      ] as const) {
        const shader = gl.createShader(type)!;
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        const ok = gl.getShaderParameter(shader, gl.COMPILE_STATUS),
          log = gl.getShaderInfoLog(shader);
        if (ok) gl.attachShader(program, shader);
        gl.deleteShader(shader);
        if (!ok) throw new Error(log || 'GPU 计算着色器编译失败');
      }
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        throw new Error(gl.getProgramInfoLog(program) || 'GPU 程序链接失败');
      return program;
    } catch (error) {
      gl.deleteProgram(program);
      throw error;
    }
  }
  private location(program: WebGLProgram, name: string) {
    let cache = this.uniforms.get(program);
    if (!cache) {
      cache = new Map();
      this.uniforms.set(program, cache);
    }
    if (!cache.has(name))
      cache.set(name, this.gl.getUniformLocation(program, name));
    return cache.get(name)!;
  }
  private run(
    name: string,
    target: Target,
    inputs: Record<string, Target> = {},
    values: Record<string, number | number[]> = {},
  ) {
    const gl = this.gl,
      program = this.programs.get(name)!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
    const particlePass = [
      'predict',
      'lambda',
      'correct',
      'velocity',
      'viscosity',
      'divergenceFactor',
      'divergenceResidual',
      'divergenceProject',
      'geometry',
      'reorder',
    ].includes(name);
    const sortPass = name === 'key' || name === 'sort' || name === 'sortMerge';
    const height = particlePass
      ? Math.max(1, Math.ceil(this.count / PARTICLE_WIDTH))
      : sortPass
        ? this.sortCount / PARTICLE_WIDTH
        : target.height;
    if (target.width !== this.viewportWidth || height !== this.viewportHeight) {
      gl.viewport(0, 0, target.width, height);
      this.viewportWidth = target.width;
      this.viewportHeight = height;
    }
    if (!this.computeActive) {
      gl.bindVertexArray(this.vao);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      this.computeActive = true;
    }
    if (program !== this.activeProgram) {
      gl.useProgram(program);
      this.activeProgram = program;
    }
    let valuesCache = this.scalarValues.get(program);
    if (!valuesCache) this.scalarValues.set(program, (valuesCache = new Map()));
    const scalar = (name: string, value: number, integer = false) => {
      if (valuesCache.get(name) === value) return;
      valuesCache.set(name, value);
      const location = this.location(program, name);
      if (integer) gl.uniform1i(location, value);
      else gl.uniform1f(location, value);
    };
    // All physics programs see the same rigid-body state. Unused samplers are optimized out.
    if (!name.startsWith('duck') && !Object.hasOwn(inputs, 'duckState'))
      inputs = { ...inputs, duckState: this.duckState };
    let unit = 0;
    for (const [key, input] of Object.entries(inputs)) {
      if (input === target) throw new Error('GPU 流体缓冲读写冲突');
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, input.texture);
      scalar(key, unit++, true);
    }
    scalar('count', this.count, true);
    scalar('particleScale', Math.cbrt(10000 / this.quality));
    scalar('duckEnabled', 1);
    scalar('sortCount', this.sortCount, true);
    scalar('initialCount', this.quality, true);
    for (const [key, value] of Object.entries(values)) {
      const location = this.location(program, key);
      if (Array.isArray(value)) {
        if (value.length === 2) gl.uniform2fv(location, value);
        if (value.length === 3) gl.uniform3fv(location, value);
        if (value.length === 4) gl.uniform4fv(location, value);
      } else
        scalar(key, value, ['stage', 'stride', 'previousCount'].includes(key));
    }
    if (name === 'volume') {
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.drawArraysInstanced(
        gl.TRIANGLES,
        0,
        6,
        this.count * GPU_SLICES_PER_PARTICLE,
      );
      gl.disable(gl.BLEND);
    } else gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  private reset() {
    this.count = this.quality;
    this.time = 0;
    this.accumulator = 0;
    this.pendingPour = 0;
    this.shakeUntil = 0;
    this.splash = [0, 0, 0];
    this.run('duckInitialize', this.duckState);
    this.run('initialize', this.position);
    const gl = this.gl;
    for (const t of [this.velocity, this.velocityTemp]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.framebuffer);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    this.dirty = true;
    this.densityDirty = true;
  }
  private runMRT(
    name: string,
    outputs: Target[],
    inputs: Record<string, Target>,
    values: Record<string, number | number[]> = {},
  ) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, outputs[0].framebuffer);
    outputs
      .slice(1)
      .forEach((t, i) =>
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0 + i + 1,
          gl.TEXTURE_2D,
          t.texture,
          0,
        ),
      );
    gl.drawBuffers(outputs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    this.run(name, outputs[0], inputs, values);
    outputs
      .slice(1)
      .forEach((_, i) =>
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0 + i + 1,
          gl.TEXTURE_2D,
          null,
          0,
        ),
      );
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  }
  private buildGrid(p: Target) {
    this.sortCount = this.count > 16384 ? 32768 : 16384;
    this.run('key', this.keys, { positions: p });
    // Bitonic sorting is bounded by texture capacity, independent of occupied-cell density.
    for (let stage = 2; stage <= this.sortCount; stage *= 2)
      for (let stride = stage / 2; stride >= 1; stride /= 2) {
        this.run(
          stride === 4 ? 'sortMerge' : 'sort',
          this.keysTemp,
          { sortedKeys: this.keys },
          { stage, stride },
        );
        [this.keys, this.keysTemp] = [this.keysTemp, this.keys];
        if (stride === 4) break;
      }
    this.run('ranges', this.ranges, { sortedKeys: this.keys });
  }
  private reorderState(predicted: boolean, previousCount: number) {
    const gl = this.gl;
    const source = predicted ? this.predicted : this.position;
    const outputs = [this.sortedPosition, this.sortedOld, this.sortedVelocity];
    // Targets rotate with ping-pong state, so configure and restore MRT attachments per pass.
    if (
      outputs.some((target) =>
        [source, this.position, this.velocity].includes(target),
      )
    )
      throw new Error('GPU 重排缓冲读写冲突');
    gl.bindFramebuffer(gl.FRAMEBUFFER, outputs[0].framebuffer);
    outputs
      .slice(1)
      .forEach((target, i) =>
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0 + i + 1,
          gl.TEXTURE_2D,
          target.texture,
          0,
        ),
      );
    gl.drawBuffers(outputs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    this.run(
      'reorder',
      outputs[0],
      {
        positions: source,
        oldPositions: this.position,
        velocities: this.velocity,
        sortedKeys: this.keys,
      },
      { previousCount },
    );
    for (let i = 1; i < 3; i++)
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0 + i,
        gl.TEXTURE_2D,
        null,
        0,
      );
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    if (predicted)
      [this.predicted, this.sortedPosition] = [
        this.sortedPosition,
        this.predicted,
      ];
    else
      [this.position, this.sortedPosition] = [
        this.sortedPosition,
        this.position,
      ];
    [this.velocity, this.sortedVelocity] = [this.sortedVelocity, this.velocity];
  }
  update(
    job: Pick<
      FluidJob,
      'elapsed' | 'speed' | 'paused' | 'forces' | 'brush' | 'particles'
    > & { actions: GpuFluidAction[] },
  ) {
    // The scene renderer shares the context. Invalidate only GL state at this
    // boundary; uniforms belong to our own programs and persist across frames.
    this.activeProgram = null;
    this.computeActive = false;
    this.viewportWidth = this.viewportHeight = 0;
    for (const action of job.actions) this.action(action);
    // A slow frame must not trigger three expensive catch-up steps and make
    // the next frame slower again. Keep the stable dt; discard excess backlog.
    const stepBudget = job.elapsed > 1 / 45 ? 1 : 2;
    if (job.paused) this.accumulator = 0;
    else
      this.accumulator = Math.min(
        stepBudget / 60,
        this.accumulator + Math.max(0, Math.min(0.1, job.elapsed)) * job.speed,
      );
    while (this.accumulator + 1e-8 >= 1 / 60) {
      this.run(
        'duckPredict',
        this.duckTemp,
        { duckState: this.duckState },
        { dt: 1 / 60, gravity: job.forces.gravity },
      );
      [this.duckState, this.duckTemp] = [this.duckTemp, this.duckState];
      const previousCount = this.count,
        b = job.brush;
      if (b?.mode === 'pour') {
        this.pendingPour = Math.min(2000, this.pendingPour + 14);
        this.pourAt = [b.x, b.z];
      }
      const added = Math.min(18, this.pendingPour, CAPACITY - this.count);
      this.count += added;
      this.pendingPour -= added;
      if (this.count === CAPACITY) this.pendingPour = 0;
      this.run(
        'predict',
        this.predicted,
        { positions: this.position, velocities: this.velocity },
        {
          dt: 1 / 60,
          time: this.time,
          ...job.forces,
          previousCount,
          pourAt: this.pourAt,
          shake:
            this.time < this.shakeUntil
              ? Math.sin((this.shakeUntil - this.time) * 16) * 22
              : 0,
          brush:
            b?.mode === 'stir' ? [b.x, b.y, b.z, b.strength] : [0, 0, 0, 0],
          brushVelocity: b ? [b.dx, b.dz] : [0, 0],
          splash: this.splash,
        },
      );
      this.splash = [0, 0, 0];
      this.buildGrid(this.predicted);
      this.reorderState(true, previousCount);
      const neighborInputs = { sortedKeys: this.keys, cellRanges: this.ranges };
      // Solid support adds constraints at the floor and walls; both quality
      // levels need three density iterations to keep compression controlled.
      for (let iteration = 0; iteration < 3; iteration++) {
        this.run('lambda', this.lambda, {
          positions: this.predicted,
          ...neighborInputs,
        });
        this.runMRT(
          'correct',
          [this.correction, ...this.reactions[1]],
          {
            positions: this.predicted,
            lambdas: this.lambda,
            ...neighborInputs,
            linearSource: this.reactions[0][0],
            angularSource: this.reactions[0][1],
          },
          { reactionReset: +(iteration === 0) },
        );
        this.reactions.reverse();
        [this.predicted, this.correction] = [this.correction, this.predicted];
      }
      this.run(
        'velocity',
        this.velocityTemp,
        { positions: this.predicted, oldPositions: this.sortedOld },
        { dt: 1 / 60, previousCount },
      );
      this.run(
        'viscosity',
        this.velocity,
        {
          positions: this.predicted,
          velocities: this.velocityTemp,
          ...neighborInputs,
        },
        { viscosity: job.forces.viscosity },
      );
      this.run('divergenceFactor', this.divergenceFactor, {
        positions: this.predicted,
        ...neighborInputs,
      });
      for (let iteration = 0; iteration < 2; iteration++) {
        this.run('divergenceResidual', this.lambda, {
          positions: this.predicted,
          velocities: this.velocity,
          factors: this.divergenceFactor,
          ...neighborInputs,
        });
        this.runMRT(
          'divergenceProject',
          [this.velocityTemp, ...this.reactions[1]],
          {
            positions: this.predicted,
            velocities: this.velocity,
            lambdas: this.lambda,
            ...neighborInputs,
            linearSource: this.reactions[0][0],
            angularSource: this.reactions[0][1],
          },
        );
        this.reactions.reverse();
        [this.velocity, this.velocityTemp] = [this.velocityTemp, this.velocity];
      }
      let reaction = this.reactions[0];
      for (const pair of this.reduction) {
        this.runMRT(
          'duckReduce',
          pair,
          { linearSource: reaction[0], angularSource: reaction[1] },
          { firstLevel: +(reaction === this.reactions[0]) },
        );
        reaction = pair;
      }
      this.run('duckIntegrate', this.duckTemp, {
        duckState: this.duckState,
        linearSource: reaction[0],
        angularSource: reaction[1],
      });
      [this.duckState, this.duckTemp] = [this.duckTemp, this.duckState];
      [this.position, this.predicted] = [this.predicted, this.position];
      this.accumulator -= 1 / 60;
      this.time += 1 / 60;
      this.dirty = true;
      this.densityDirty = false;
    }
    if (this.dirty && !job.particles) {
      if (this.densityDirty) {
        this.buildGrid(this.position);
        this.reorderState(false, this.count);
        this.densityDirty = false;
      }
      this.run('geometry', this.geometry[0], {
        positions: this.position,
        sortedKeys: this.keys,
        cellRanges: this.ranges,
      });
      let source = this.position;
      for (const target of this.bounds) {
        this.run(
          'bounds',
          target,
          { source },
          { firstLevel: +(source === this.position) },
        );
        source = target;
      }
      this.run('volume', this.atlas, {
        positions: this.geometry[0],
        metric0: this.geometry[1],
        metric1: this.geometry[2],
        metric2: this.geometry[3],
      });
      this.run(
        'surfaceFilter',
        this.surfaceTemp,
        { source: this.atlas, guide: this.atlas },
        { axis: [1, 0, 0] },
      );
      this.run(
        'surfaceFilter',
        this.surfaceFiltered,
        { source: this.surfaceTemp, guide: this.atlas },
        { axis: [0, 1, 0] },
      );
      this.run(
        'surfaceFilter',
        this.surfaceTemp,
        { source: this.surfaceFiltered, guide: this.atlas },
        { axis: [0, 0, 1] },
      );
      this.dirty = false;
    }
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }
  private action(action: GpuFluidAction) {
    if (action.type === 'quality') {
      if (action.count !== 15000 && action.count !== 30000)
        throw new Error('粒子精度须为 15000 或 30000');
      this.quality = action.count;
      this.reset();
    }
    if (action.type === 'reset') this.reset();
    if (action.type === 'drain') {
      this.count = Math.max(0, this.count - (action.amount ?? 500));
      this.dirty = true;
      this.densityDirty = true;
    }
    if (action.type === 'pour') {
      this.pendingPour = Math.min(
        2000,
        this.pendingPour + (action.amount ?? 500),
      );
      this.pourAt = [action.x ?? -0.7, action.z ?? 0];
    }
    if (action.type === 'splash')
      this.splash = [action.x ?? 0, action.z ?? 0, action.strength ?? 1];
    if (action.type === 'shake') this.shakeUntil = this.time + 0.8;
  }
  destroy() {
    for (const t of this.targets) {
      this.gl.deleteTexture(t.texture);
      this.gl.deleteFramebuffer(t.framebuffer);
    }
    for (const program of this.programs.values())
      this.gl.deleteProgram(program);
    this.gl.deleteVertexArray(this.vao);
  }
}
