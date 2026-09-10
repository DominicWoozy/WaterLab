import { loadDuckModel } from './duck-model';
import { GPU_VOLUME_SIZE } from './gpu-volume-config';
import {
  VOLUME_MIN,
  VOLUME_MAX,
  SURFACE_DENSITY,
  ABSORPTION,
} from './fluid-volume';
import { GpuFluid, type GpuFluidAction } from './gpu-fluid';
import type { ParticleQuality } from './gpu-particle-config';
import {
  fullscreenVertex,
  particleVertex,
  particleFragment,
  surfaceFragment,
} from './water-shaders';
export type WaterSettings = {
  strength: number;
  speed: number;
  viscosity: number;
  gravity: number;
  agitation: number;
  light: number;
  reflection: boolean;
  caustics: boolean;
  particles: boolean;
  details: boolean;
  surfaceTension: boolean;
  paused: boolean;
  mode: 'stir' | 'pour' | 'orbit';
};
export const defaults: WaterSettings = {
  strength: 1,
  speed: 1,
  viscosity: 0.025,
  gravity: 9.8,
  agitation: 0,
  light: 1.3,
  reflection: true,
  caustics: true,
  particles: false,
  details: true,
  surfaceTension: true,
  paused: false,
  mode: 'stir',
};
export type WaterStats = {
  fps: number;
  count: number;
  quality?: ParticleQuality;
  capacity?: number;
  backend?: 'WebGPU' | 'WebGL2';
};
export function createWebGLWater(
  canvas: HTMLCanvasElement,
  getSettings: () => WaterSettings,
  onStats: (stats: WaterStats) => void,
  onError: (message: string) => void,
) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    powerPreference: 'high-performance',
  });
  if (!gl)
    throw new Error('水体合成需要 WebGL 2 支持，请启用浏览器硬件加速后重试。');

  const fluid = new GpuFluid(gl);
  const duck = loadDuckModel(gl, onError);
  const actions: GpuFluidAction[] = [];
  let lastUpdate = 0;
  const programs: WebGLProgram[] = [];
  const createProgram = (vertex: string, fragment: string) => {
    const program = gl.createProgram()!;
    for (const [type, source] of [
      [gl.VERTEX_SHADER, vertex],
      [gl.FRAGMENT_SHADER, fragment],
    ] as const) {
      const shader = gl.createShader(type)!;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const error = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        gl.deleteProgram(program);
        throw new Error(error || '水体着色器编译失败');
      }
      gl.attachShader(program, shader);
      gl.deleteShader(shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(program) || '水体着色器链接失败');
    programs.push(program);
    return program;
  };
  const particles = createProgram(particleVertex, particleFragment),
    surface = createProgram(fullscreenVertex, surfaceFragment);
  const quadBuffer = gl.createBuffer()!;
  const particleVAO = gl.createVertexArray()!,
    quadVAOs: WebGLVertexArrayObject[] = [];
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW,
  );
  for (const program of [surface]) {
    const vao = gl.createVertexArray()!;
    quadVAOs.push(vao);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    const loc = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }
  const uniformCache = new Map<
    WebGLProgram,
    Map<string, WebGLUniformLocation | null>
  >();
  const uniform = (program: WebGLProgram, name: string) => {
    if (!uniformCache.has(program)) uniformCache.set(program, new Map());
    const cache = uniformCache.get(program)!;
    if (!cache.has(name)) cache.set(name, gl.getUniformLocation(program, name));
    return cache.get(name)!;
  };
  const f = (program: WebGLProgram, name: string, value: number) =>
    gl.uniform1f(uniform(program, name), value);
  let yaw = 0.58,
    pitch = 0.49,
    zoom = 8,
    raf = 0,
    frames = 0,
    statTime = performance.now(),
    disposed = false,
    lostContext = false,
    renderScale = 1;
  let resizePending = true;
  const normalize = (v: number[]) => {
    const length = Math.hypot(...v);
    return v.map((n) => n / length);
  };
  const cross = (a: number[], b: number[]) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const camera = () => {
    const eye = [
      Math.sin(yaw) * Math.cos(pitch) * zoom,
      Math.sin(pitch) * zoom,
      Math.cos(yaw) * Math.cos(pitch) * zoom,
    ];
    const forward = normalize([-eye[0], -0.05 - eye[1], -eye[2]]),
      right = normalize(cross(forward, [0, 1, 0]));
    return { eye, forward, right, up: cross(right, forward) };
  };
  const cameraUniforms = (program: WebGLProgram) => {
    const c = camera();
    gl.uniform2f(uniform(program, 'resolution'), canvas.width, canvas.height);
    gl.uniform3fv(uniform(program, 'eye'), c.eye);
    gl.uniform3fv(uniform(program, 'cameraRight'), c.right);
    gl.uniform3fv(uniform(program, 'cameraUp'), c.up);
    gl.uniform3fv(uniform(program, 'cameraForward'), c.forward);
    f(program, 'offsetX', canvas.width / canvas.height > 1.18 ? 0.19 : 0);
  };
  const resize = () => {
    const rect = canvas.getBoundingClientRect(),
      scale = Math.min(
        window.devicePixelRatio,
        1.25,
        (1250 * renderScale) / Math.max(1, rect.width),
      );
    const width = Math.max(1, Math.round(rect.width * scale));
    const height = Math.max(1, Math.round(rect.height * scale));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    gl.viewport(0, 0, canvas.width, canvas.height);
  };
  const observer = new ResizeObserver(() => {
    resizePending = true;
  });
  observer.observe(canvas);
  let pointer: {
    id: number;
    x: number;
    y: number;
    lastX: number;
    lastY: number;
    orbit: boolean;
    moved: boolean;
    world: number[];
    dx: number;
    dz: number;
  } | null = null;
  const locate = (x: number, y: number) => {
    const rect = canvas.getBoundingClientRect();
    let u = (x - rect.left - rect.width * 0.5) / rect.height;
    if (rect.width / rect.height > 1.18) u += 0.19;
    const v = -(y - rect.top - rect.height * 0.5) / rect.height;
    const c = camera(),
      d = normalize(
        c.forward.map((n, i) => n * 1.55 + u * c.right[i] + v * c.up[i]),
      );
    const t = (-0.25 - c.eye[1]) / d[1];
    return [
      Math.max(-1.7, Math.min(1.7, c.eye[0] + d[0] * t)),
      -0.25,
      Math.max(-1.2, Math.min(1.2, c.eye[2] + d[2] * t)),
    ];
  };
  const render = (now: number) => {
    if (disposed || lostContext) return;
    if (resizePending) {
      resize();
      resizePending = false;
    }
    const s = getSettings();
    const b = pointer && !pointer.orbit ? pointer : null;
    try {
      fluid.update({
        elapsed: document.hidden
          ? 0
          : lastUpdate
            ? Math.min(0.05, (now - lastUpdate) / 1000)
            : 1 / 60,
        speed: s.speed,
        paused: s.paused || document.hidden,
        particles: s.particles,
        forces: {
          gravity: s.gravity,
          viscosity: s.viscosity,
          agitation: s.agitation,
        },
        brush:
          b && s.mode !== 'orbit'
            ? {
                x: b.world[0],
                y: b.world[1],
                z: b.world[2],
                dx: b.dx,
                dz: b.dz,
                strength: s.strength,
                mode: s.mode,
              }
            : undefined,
        actions: actions.splice(0),
      });
      lastUpdate = now;
    } catch (error) {
      lostContext = true;
      onError(error instanceof Error ? error.message : 'GPU 流体计算失败');
      return;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(surface);
    gl.bindVertexArray(quadVAOs[0]);
    cameraUniforms(surface);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, fluid.volume);
    gl.uniform1i(uniform(surface, 'densityVolume'), 0);
    gl.uniform3fv(uniform(surface, 'volumeMin'), VOLUME_MIN);
    gl.uniform3fv(uniform(surface, 'volumeMax'), VOLUME_MAX);
    gl.uniform3fv(uniform(surface, 'volumeSize'), GPU_VOLUME_SIZE);
    gl.uniform3fv(uniform(surface, 'absorption'), ABSORPTION);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, fluid.volumeBounds);
    gl.uniform1i(uniform(surface, 'waterBounds'), 1);
    for (const [index, name, texture] of [
      [2, 'duckState', fluid.duck],
      [3, 'duckBVH', duck.bvh],
      [4, 'duckTriangles', duck.triangles],
      [5, 'duckAlbedo', duck.albedo],
    ] as const) {
      gl.activeTexture(gl.TEXTURE0 + index);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.uniform1i(uniform(surface, name), index);
    }
    f(surface, 'duckReady', +duck.ready);
    f(surface, 'isoDensity', SURFACE_DENSITY);
    f(surface, 'time', fluid.time);
    f(surface, 'lightPower', s.light);
    f(surface, 'reflectionOn', +s.reflection);
    f(surface, 'causticsOn', +s.caustics);
    f(surface, 'particleView', +s.particles);
    f(surface, 'brushOn', +(!!pointer && !pointer.orbit));
    gl.uniform3fv(uniform(surface, 'brush'), pointer?.world || [0, -0.25, 0]);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.ALWAYS);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disable(gl.DEPTH_TEST);
    if (s.particles) {
      gl.useProgram(particles);
      cameraUniforms(particles);
      gl.bindVertexArray(particleVAO);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fluid.positions);
      gl.uniform1i(uniform(particles, 'positions'), 0);
      f(particles, 'radius', 0.021);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.drawArrays(gl.POINTS, 0, fluid.count);
      gl.disable(gl.DEPTH_TEST);
    }
    frames++;
    if (now - statTime > 1000) {
      onStats({
        fps: Math.round((frames * 1000) / (now - statTime)),
        count: fluid.count,
        quality: fluid.quality,
        capacity: 30000,
        backend: 'WebGL2',
      });
      const measured = (frames * 1000) / (now - statTime);
      const nextScale =
        measured < 35
          ? Math.max(0.65, renderScale * 0.9)
          : measured > 56
            ? Math.min(1.12, renderScale * 1.03)
            : renderScale;
      if (Math.abs(nextScale - renderScale) > 0.015) {
        renderScale = nextScale;
        resizePending = true;
      }
      frames = 0;
      statTime = now;
    }
    raf = requestAnimationFrame(render);
  };
  const down = (e: PointerEvent) => {
    if (e.button !== 0 && e.button !== 2) return;
    const s = getSettings();
    pointer = {
      id: e.pointerId,
      x: e.clientX,
      y: e.clientY,
      lastX: e.clientX,
      lastY: e.clientY,
      orbit: s.mode === 'orbit' || e.shiftKey || e.button === 2,
      moved: false,
      world: locate(e.clientX, e.clientY),
      dx: 0,
      dz: 0,
    };
    canvas.setPointerCapture(e.pointerId);
  };
  const move = (e: PointerEvent) => {
    if (!pointer || pointer.id !== e.pointerId) return;
    const dx = e.clientX - pointer.lastX,
      dy = e.clientY - pointer.lastY;
    if (Math.hypot(e.clientX - pointer.x, e.clientY - pointer.y) > 5)
      pointer.moved = true;
    if (pointer.orbit) {
      yaw -= dx * 0.007;
      pitch = Math.max(0.14, Math.min(1.3, pitch + dy * 0.006));
    } else {
      const next = locate(e.clientX, e.clientY);
      pointer.dx = Math.max(-6, Math.min(6, (next[0] - pointer.world[0]) * 35));
      pointer.dz = Math.max(-6, Math.min(6, (next[2] - pointer.world[2]) * 35));
      pointer.world = next;
    }
    pointer.lastX = e.clientX;
    pointer.lastY = e.clientY;
  };
  const up = (e: PointerEvent) => {
    if (!pointer || pointer.id !== e.pointerId) return;
    if (!pointer.orbit && !pointer.moved && !getSettings().paused) {
      if (getSettings().mode === 'pour')
        actions.push({
          type: 'pour',
          x: pointer.world[0],
          z: pointer.world[2],
          amount: 100,
        });
      else
        actions.push({
          type: 'splash',
          x: pointer.world[0],
          z: pointer.world[2],
          strength: getSettings().strength,
        });
    }
    pointer = null;
  };
  const cancel = () => {
      pointer = null;
    },
    contextMenu = (e: Event) => e.preventDefault();
  const wheel = (e: WheelEvent) => {
    e.preventDefault();
    zoom = Math.max(5.8, Math.min(12, zoom + e.deltaY * 0.006));
  };
  const key = (e: KeyboardEvent) => {
    if (
      [
        'ArrowLeft',
        'ArrowRight',
        'ArrowUp',
        'ArrowDown',
        '+',
        '-',
        'Enter',
        ' ',
      ].includes(e.key)
    )
      e.preventDefault();
    if (e.key === 'ArrowLeft') yaw -= 0.1;
    if (e.key === 'ArrowRight') yaw += 0.1;
    if (e.key === 'ArrowUp') pitch = Math.min(1.3, pitch + 0.08);
    if (e.key === 'ArrowDown') pitch = Math.max(0.14, pitch - 0.08);
    if (e.key === '+') zoom = Math.max(5.8, zoom - 0.3);
    if (e.key === '-') zoom = Math.min(12, zoom + 0.3);
    if ((e.key === 'Enter' || e.key === ' ') && !getSettings().paused)
      actions.push({ type: 'splash', strength: getSettings().strength });
  };
  const lost = (e: Event) => {
    e.preventDefault();
    lostContext = true;
    cancelAnimationFrame(raf);
    onError('图形上下文已断开，请重新加载恢复水体。');
  };
  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', cancel);
  canvas.addEventListener('wheel', wheel, { passive: false });
  canvas.addEventListener('keydown', key);
  canvas.addEventListener('contextmenu', contextMenu);
  canvas.addEventListener('webglcontextlost', lost);
  raf = requestAnimationFrame(render);
  return {
    setQuality: (count: ParticleQuality) => {
      if (count > 30000) count = 30000;
      actions.length = 0;
      actions.push({ type: 'quality', count });
      pointer = null;
    },
    ripple: () => {
      actions.push({ type: 'splash', strength: getSettings().strength });
    },
    shake: () => {
      actions.push({ type: 'shake' });
    },
    pour: () => {
      actions.push({ type: 'pour', amount: 500 });
    },
    drain: () => {
      actions.push({ type: 'drain', amount: 500 });
    },
    reset: () => {
      actions.length = 0;
      actions.push({ type: 'reset' });
      pointer = null;
      yaw = 0.58;
      pitch = 0.49;
      zoom = 8;
    },
    destroy: () => {
      disposed = true;
      cancelAnimationFrame(raf);
      observer.disconnect();
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', up);
      canvas.removeEventListener('pointercancel', cancel);
      canvas.removeEventListener('wheel', wheel);
      canvas.removeEventListener('keydown', key);
      canvas.removeEventListener('contextmenu', contextMenu);
      canvas.removeEventListener('webglcontextlost', lost);
      duck.destroy();
      fluid.destroy();
      gl.deleteBuffer(quadBuffer);
      gl.deleteVertexArray(particleVAO);
      quadVAOs.forEach((v) => gl.deleteVertexArray(v));
      programs.forEach((p) => gl.deleteProgram(p));
    },
  };
}

/** Prepare the full WebGPU backend before acquiring the canvas context, so a
 * capability/compilation failure can still fall back to WebGL2 on the same canvas. */
export function createWater(
  canvas: HTMLCanvasElement,
  getSettings: () => WaterSettings,
  onStats: (s: WaterStats) => void,
  onError: (s: string) => void,
) {
  let engine: ReturnType<typeof createWebGLWater> | null = null,
    disposed = false;
  const pending: Array<(e: ReturnType<typeof createWebGLWater>) => void> = [];
  void (async () => {
    if (
      navigator.gpu &&
      new URLSearchParams(location.search).get('backend') !== 'webgl'
    ) {
      let device: GPUDevice | undefined;
      try {
        const adapter = await navigator.gpu.requestAdapter({
          powerPreference: 'high-performance',
        });
        if (!adapter) throw new Error('WebGPU adapter unavailable');
        const features: GPUFeatureName[] = [];
        for (const feature of ['float32-filterable'] as const)
          if (adapter.features.has(feature)) features.push(feature);
        device = await adapter.requestDevice({ requiredFeatures: features });
        const { createWebGPUWater } = await import('./webgpu/engine');
        if (disposed) {
          device.destroy();
          return;
        }
        engine = await createWebGPUWater(
          canvas,
          getSettings,
          onStats,
          onError,
          device,
        );
      } catch (error) {
        device?.destroy();
        console.warn('WebGPU initialization failed; using WebGL2', error);
      }
    }
    if (disposed) {
      engine?.destroy();
      return;
    }
    if (!engine) {
      engine = createWebGLWater(canvas, getSettings, onStats, onError);
      onStats({
        fps: 0,
        count: 15000,
        quality: 15000,
        capacity: 30000,
        backend: 'WebGL2',
      });
    }
    pending.splice(0).forEach((fn) => fn(engine!));
  })().catch((error) => {
    if (!disposed)
      onError(error instanceof Error ? error.message : '无法启动 GPU 引擎');
  });
  const call = (fn: (e: ReturnType<typeof createWebGLWater>) => void) => {
    if (disposed) return;
    if (engine) fn(engine);
    else pending.push(fn);
  };
  return {
    setQuality: (count: ParticleQuality) => call((e) => e.setQuality(count)),
    ripple: () => call((e) => e.ripple()),
    shake: () => call((e) => e.shake()),
    pour: () => call((e) => e.pour()),
    drain: () => call((e) => e.drain()),
    reset: () => call((e) => e.reset()),
    destroy: () => {
      disposed = true;
      pending.length = 0;
      engine?.destroy();
    },
  };
}
