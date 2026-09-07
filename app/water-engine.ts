import {
  VOLUME_SIZE,
  VOLUME_MIN,
  VOLUME_MAX,
  SURFACE_DENSITY,
  ABSORPTION,
} from './fluid-volume';
import { CAPACITY, DEFAULT_COUNT } from './fluid-simulation';
import type { FluidAction, FluidFrame, FluidJob } from './fluid-runtime';
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
  paused: false,
  mode: 'stir',
};
export type WaterStats = { fps: number; count: number };
export function createWater(
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

  const makeVolumeTexture = () => {
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_3D, texture);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const axis of [
      gl.TEXTURE_WRAP_S,
      gl.TEXTURE_WRAP_T,
      gl.TEXTURE_WRAP_R,
    ])
      gl.texParameteri(gl.TEXTURE_3D, axis, gl.CLAMP_TO_EDGE);
    gl.texImage3D(
      gl.TEXTURE_3D,
      0,
      gl.R16F,
      ...VOLUME_SIZE,
      0,
      gl.RED,
      gl.FLOAT,
      null,
    );
    return texture;
  };
  let currentVolume = makeVolumeTexture(),
    previousVolume = makeVolumeTexture();
  let simTime = 0,
    particleCount = DEFAULT_COUNT,
    volumeTop = 0,
    previousTop = 0,
    snapshotTime = 0,
    snapshotInterval = 33,
    hasVolume = false;
  let workerBusy = false,
    recycleVolume: ArrayBuffer | undefined,
    recyclePositions: ArrayBuffer | undefined;
  let lastRequest = 0,
    lastJobParticles: boolean | null = null,
    positionsReady = false;
  const actions: FluidAction[] = [];
  const worker = new Worker(new URL('./fluid-worker.ts', import.meta.url), {
    type: 'module',
  });
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
  const particleBuffer = gl.createBuffer()!,
    quadBuffer = gl.createBuffer()!;
  const particleVAO = gl.createVertexArray()!,
    quadVAOs: WebGLVertexArrayObject[] = [];
  gl.bindVertexArray(particleVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, particleBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, CAPACITY * 3 * 4, gl.DYNAMIC_DRAW);
  const position = gl.getAttribLocation(particles, 'position');
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
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
    canvas.width = Math.max(1, Math.round(rect.width * scale));
    canvas.height = Math.max(1, Math.round(rect.height * scale));
    gl.viewport(0, 0, canvas.width, canvas.height);
  };
  resize();
  const observer = new ResizeObserver(() => {
    try {
      resize();
    } catch (e) {
      onError(e instanceof Error ? e.message : '水体画布调整失败');
    }
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
  const submitJob = (now: number) => {
    if (workerBusy || disposed || lostContext || document.hidden) return;
    const s = getSettings();
    if (
      s.paused &&
      hasVolume &&
      actions.length === 0 &&
      s.particles === lastJobParticles
    )
      return;
    if (lastRequest && now - lastRequest < 16 && actions.length === 0) return;
    const b = pointer && !pointer.orbit ? pointer : null;
    const job: FluidJob = {
      elapsed: lastRequest
        ? Math.min(0.05, (now - lastRequest) / 1000)
        : 1 / 60,
      speed: s.speed,
      paused: s.paused,
      forces: {
        gravity: s.gravity,
        viscosity: s.viscosity,
        agitation: s.agitation,
      },
      particles: s.particles,
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
      recycleVolume,
      recyclePositions,
    };
    const transfers: Transferable[] = [];
    if (recycleVolume) transfers.push(recycleVolume);
    if (recyclePositions) transfers.push(recyclePositions);
    recycleVolume = undefined;
    recyclePositions = undefined;
    workerBusy = true;
    lastRequest = now;
    lastJobParticles = s.particles;
    worker.postMessage(job, transfers);
  };
  worker.onmessage = (
    event: MessageEvent<FluidFrame & { type: string; message?: string }>,
  ) => {
    workerBusy = false;
    if (disposed || lostContext) return;
    const frame = event.data;
    if (frame.type === 'error') {
      onError(frame.message || '后台水体计算失败');
      lostContext = true;
      return;
    }
    const now = performance.now();
    simTime = frame.time;
    particleCount = frame.count;
    if (frame.volume) {
      const data = new Float32Array(frame.volume);
      [previousVolume, currentVolume] = [currentVolume, previousVolume];
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_3D, currentVolume);
      gl.texSubImage3D(
        gl.TEXTURE_3D,
        0,
        0,
        0,
        0,
        ...VOLUME_SIZE,
        gl.RED,
        gl.FLOAT,
        data,
      );
      if (!hasVolume) {
        gl.bindTexture(gl.TEXTURE_3D, previousVolume);
        gl.texSubImage3D(
          gl.TEXTURE_3D,
          0,
          0,
          0,
          0,
          ...VOLUME_SIZE,
          gl.RED,
          gl.FLOAT,
          data,
        );
      }
      previousTop = hasVolume ? volumeTop : frame.top;
      volumeTop = frame.top;
      snapshotInterval = snapshotTime
        ? Math.max(16, Math.min(100, now - snapshotTime))
        : 33;
      snapshotTime = now;
      hasVolume = true;
      recycleVolume = frame.volume;
    }
    if (frame.positions) {
      positionsReady = true;
      gl.bindBuffer(gl.ARRAY_BUFFER, particleBuffer);
      gl.bufferSubData(
        gl.ARRAY_BUFFER,
        0,
        new Float32Array(frame.positions, 0, frame.count * 3),
      );
      recyclePositions = frame.positions;
    }
    // Run independently of RAF while leaving the main event loop available for input.
    if (!getSettings().paused)
      setTimeout(() => submitJob(performance.now()), 0);
  };
  worker.onerror = () => {
    workerBusy = false;
    lostContext = true;
    onError('后台流体线程未能启动，请重新加载页面。');
  };
  const render = (now: number) => {
    if (disposed || lostContext) return;
    const s = getSettings();
    submitJob(now);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(surface);
    gl.bindVertexArray(quadVAOs[0]);
    cameraUniforms(surface);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, currentVolume);
    gl.uniform1i(uniform(surface, 'densityVolume'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, previousVolume);
    gl.uniform1i(uniform(surface, 'previousVolume'), 1);
    f(
      surface,
      'fieldBlend',
      Math.min(1, (now - snapshotTime) / snapshotInterval),
    );
    gl.uniform3fv(uniform(surface, 'volumeMin'), VOLUME_MIN);
    gl.uniform3fv(uniform(surface, 'volumeMax'), VOLUME_MAX);
    gl.uniform3fv(uniform(surface, 'volumeSize'), VOLUME_SIZE);
    gl.uniform3fv(uniform(surface, 'absorption'), ABSORPTION);
    f(surface, 'volumeTop', Math.max(previousTop, volumeTop));
    f(surface, 'isoDensity', SURFACE_DENSITY);
    f(surface, 'time', simTime);
    f(surface, 'lightPower', s.light);
    f(surface, 'reflectionOn', +s.reflection);
    f(surface, 'causticsOn', +s.caustics);
    f(surface, 'particleView', +s.particles);
    f(surface, 'brushOn', +(!!pointer && !pointer.orbit));
    gl.uniform3fv(uniform(surface, 'brush'), pointer?.world || [0, -0.25, 0]);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    if (s.particles && positionsReady) {
      gl.useProgram(particles);
      cameraUniforms(particles);
      gl.bindVertexArray(particleVAO);
      gl.bindBuffer(gl.ARRAY_BUFFER, particleBuffer);
      f(particles, 'radius', 0.021);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.drawArrays(gl.POINTS, 0, particleCount);
      gl.disable(gl.DEPTH_TEST);
    }
    frames++;
    if (now - statTime > 1000) {
      onStats({
        fps: Math.round((frames * 1000) / (now - statTime)),
        count: particleCount,
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
        resize();
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
      worker.terminate();
      gl.deleteTexture(currentVolume);
      gl.deleteTexture(previousVolume);
      gl.deleteBuffer(particleBuffer);
      gl.deleteBuffer(quadBuffer);
      gl.deleteVertexArray(particleVAO);
      quadVAOs.forEach((v) => gl.deleteVertexArray(v));
      programs.forEach((p) => gl.deleteProgram(p));
    },
  };
}
