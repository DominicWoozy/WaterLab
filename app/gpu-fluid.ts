import {
  GPU_ATLAS_SIZE,
  GPU_SLICES_PER_PARTICLE,
} from './gpu-volume-config.ts';
import * as shaders from './gpu-fluid-shaders.ts';
import type { FluidAction, FluidJob } from './fluid-runtime.ts';
import { CAPACITY, DEFAULT_COUNT } from './fluid-simulation.ts';
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
  private gl: WebGL2RenderingContext;
  private targets: Target[] = [];
  private programs = new Map<string, WebGLProgram>();
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
  private atlas: Target;
  private bounds: Target[] = [];
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
      this.position = this.target();
      this.predicted = this.target();
      this.correction = this.target();
      this.velocity = this.target();
      this.velocityTemp = this.target();
      this.keys = this.target();
      this.keysTemp = this.target();
      this.ranges = this.target(128, 137);
      this.lambda = this.target();
      this.atlas = this.target(...GPU_ATLAS_SIZE, true);
      for (let size = 64; size >= 1; size /= 2)
        this.bounds.push(this.target(size, size));
      for (const name of [
        'bounds',
        'initialize',
        'predict',
        'key',
        'sort',
        'ranges',
        'lambda',
        'correct',
        'velocity',
        'viscosity',
        'volume',
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
  get positions() {
    return this.position.texture;
  }
  get volume() {
    return this.atlas.texture;
  }
  get volumeBounds() {
    return this.bounds[this.bounds.length - 1].texture;
  }
  private target(width = 128, height = 128, half = false): Target {
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
    gl.viewport(0, 0, target.width, target.height);
    gl.bindVertexArray(this.vao);
    gl.useProgram(program);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    let unit = 0;
    for (const [key, input] of Object.entries(inputs)) {
      if (input === target) throw new Error('GPU 流体缓冲读写冲突');
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, input.texture);
      gl.uniform1i(this.location(program, key), unit++);
    }
    gl.uniform1i(this.location(program, 'count'), this.count);
    for (const [key, value] of Object.entries(values)) {
      const location = this.location(program, key);
      if (Array.isArray(value)) {
        if (value.length === 2) gl.uniform2fv(location, value);
        if (value.length === 3) gl.uniform3fv(location, value);
        if (value.length === 4) gl.uniform4fv(location, value);
      } else if (['stage', 'stride', 'previousCount'].includes(key))
        gl.uniform1i(location, value);
      else gl.uniform1f(location, value);
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
    this.count = DEFAULT_COUNT;
    this.time = 0;
    this.accumulator = 0;
    this.pendingPour = 0;
    this.shakeUntil = 0;
    this.splash = [0, 0, 0];
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
  private buildGrid(p: Target) {
    this.run('key', this.keys, { positions: p });
    // Bitonic sorting is bounded by texture capacity, independent of occupied-cell density.
    for (let stage = 2; stage <= 16384; stage *= 2)
      for (let stride = stage / 2; stride >= 1; stride /= 2) {
        this.run(
          'sort',
          this.keysTemp,
          { sortedKeys: this.keys },
          { stage, stride },
        );
        [this.keys, this.keysTemp] = [this.keysTemp, this.keys];
      }
    this.run('ranges', this.ranges, { sortedKeys: this.keys });
  }
  update(
    job: Pick<
      FluidJob,
      | 'elapsed'
      | 'speed'
      | 'paused'
      | 'forces'
      | 'brush'
      | 'actions'
      | 'particles'
    >,
  ) {
    for (const action of job.actions) this.action(action);
    if (job.paused) this.accumulator = 0;
    else
      this.accumulator = Math.min(
        0.05,
        this.accumulator + Math.max(0, Math.min(0.1, job.elapsed)) * job.speed,
      );
    while (this.accumulator + 1e-8 >= 1 / 60) {
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
      const neighborInputs = { sortedKeys: this.keys, cellRanges: this.ranges };
      for (let iteration = 0; iteration < 2; iteration++) {
        this.run('lambda', this.lambda, {
          positions: this.predicted,
          ...neighborInputs,
        });
        this.run('correct', this.correction, {
          positions: this.predicted,
          lambdas: this.lambda,
          ...neighborInputs,
        });
        [this.predicted, this.correction] = [this.correction, this.predicted];
      }
      this.run(
        'velocity',
        this.velocityTemp,
        { positions: this.predicted, oldPositions: this.position },
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
      [this.position, this.predicted] = [this.predicted, this.position];
      this.accumulator -= 1 / 60;
      this.time += 1 / 60;
      this.dirty = true;
      this.densityDirty = false;
    }
    if (this.dirty && !job.particles) {
      if (this.densityDirty) {
        this.buildGrid(this.position);
        this.run('lambda', this.lambda, {
          positions: this.position,
          sortedKeys: this.keys,
          cellRanges: this.ranges,
        });
        this.densityDirty = false;
      }
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
        positions: this.position,
        lambdas: this.lambda,
      });
      this.dirty = false;
    }
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }
  private action(action: FluidAction) {
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
