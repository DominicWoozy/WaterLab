import { ParticleFluid } from './fluid-simulation';
import {
  fullscreenVertex,
  particleVertex,
  particleFragment,
  blurFragment,
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
  viscosity: 0.12,
  gravity: 9.8,
  agitation: 0.4,
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
  if (!gl || !gl.getExtension('EXT_color_buffer_float'))
    throw new Error(
      '粒子水面需要 WebGL 2 和浮点渲染支持，请启用浏览器硬件加速后重试。',
    );
  const fluid = new ParticleFluid();
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
    blur = createProgram(fullscreenVertex, blurFragment),
    surface = createProgram(fullscreenVertex, surfaceFragment);
  const particleBuffer = gl.createBuffer()!,
    quadBuffer = gl.createBuffer()!;
  const particleVAO = gl.createVertexArray()!,
    quadVAOs: WebGLVertexArrayObject[] = [];
  gl.bindVertexArray(particleVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, particleBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, fluid.positions.byteLength, gl.DYNAMIC_DRAW);
  const position = gl.getAttribLocation(particles, 'position');
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW,
  );
  for (const program of [blur, surface]) {
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
  type Target = {
    texture: WebGLTexture;
    framebuffer: WebGLFramebuffer;
    depth?: WebGLRenderbuffer;
  };
  let targets: Target[] = [];
  const destroyTargets = () => {
    for (const t of targets) {
      gl.deleteTexture(t.texture);
      gl.deleteFramebuffer(t.framebuffer);
      if (t.depth) gl.deleteRenderbuffer(t.depth);
    }
    targets = [];
  };
  const makeTarget = (floating: boolean, withDepth = false): Target => {
    const texture = gl.createTexture()!,
      framebuffer = gl.createFramebuffer()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      floating ? gl.R32F : gl.RGBA8,
      canvas.width,
      canvas.height,
      0,
      floating ? gl.RED : gl.RGBA,
      floating ? gl.FLOAT : gl.UNSIGNED_BYTE,
      null,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0,
    );
    let depth: WebGLRenderbuffer | undefined;
    if (withDepth) {
      depth = gl.createRenderbuffer()!;
      gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
      gl.renderbufferStorage(
        gl.RENDERBUFFER,
        gl.DEPTH_COMPONENT24,
        canvas.width,
        canvas.height,
      );
      gl.framebufferRenderbuffer(
        gl.FRAMEBUFFER,
        gl.DEPTH_ATTACHMENT,
        gl.RENDERBUFFER,
        depth,
      );
    }
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
      throw new Error('设备无法建立水面渲染缓冲区');
    return { texture, framebuffer, depth };
  };
  let yaw = 0.58,
    pitch = 0.49,
    zoom = 8,
    raf = 0,
    previous = 0,
    accumulator = 0,
    frames = 0,
    statTime = 0,
    disposed = false,
    lostContext = false,
    burst = 0;
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
        1280 / Math.max(1, rect.width),
      );
    canvas.width = Math.max(1, Math.round(rect.width * scale));
    canvas.height = Math.max(1, Math.round(rect.height * scale));
    destroyTargets();
    targets = [
      makeTarget(true, true),
      makeTarget(true),
      makeTarget(true),
      makeTarget(false),
    ];
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
  const bindTexture = (
    program: WebGLProgram,
    name: string,
    texture: WebGLTexture,
    unit: number,
  ) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(uniform(program, name), unit);
  };
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
    const s = getSettings(),
      dt = previous ? Math.min((now - previous) / 1000, 0.05) : 0;
    previous = now;
    if (!s.paused && !document.hidden) {
      accumulator = Math.min(0.05, accumulator + dt * s.speed);
      while (accumulator >= 1 / 120) {
        if (pointer && !pointer.orbit) {
          const [x, y, z] = pointer.world;
          if (s.mode === 'pour') fluid.pour(x, z, 4);
          else fluid.stir(x, y, z, pointer.dx, pointer.dz, s.strength, 1 / 120);
          pointer.dx *= 0.97;
          pointer.dz *= 0.97;
        }
        if (burst > 0) {
          fluid.pour(-0.7, -0.05, Math.min(5, burst));
          burst -= 5;
        }
        fluid.step(1 / 120, {
          gravity: s.gravity,
          viscosity: s.viscosity,
          agitation: s.agitation,
        });
        accumulator -= 1 / 120;
      }
    } else accumulator = 0;
    gl.bindBuffer(gl.ARRAY_BUFFER, particleBuffer);
    gl.bufferSubData(
      gl.ARRAY_BUFFER,
      0,
      fluid.positions.subarray(0, fluid.count * 3),
    );
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(particles);
    cameraUniforms(particles);
    gl.bindVertexArray(particleVAO);
    f(particles, 'radius', s.particles ? 0.047 : 0.115);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targets[0].framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    f(particles, 'thicknessPass', 0);
    gl.drawArrays(gl.POINTS, 0, fluid.count);
    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targets[3].framebuffer);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    f(particles, 'thicknessPass', 1);
    gl.drawArrays(gl.POINTS, 0, fluid.count);
    gl.disable(gl.BLEND);
    let depthTexture = targets[0].texture;
    if (!s.particles) {
      gl.useProgram(blur);
      gl.bindVertexArray(quadVAOs[0]);
      gl.uniform2f(uniform(blur, 'resolution'), canvas.width, canvas.height);
      for (let i = 0; i < 6; i++) {
        const target = targets[1 + (i % 2)];
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
        bindTexture(blur, 'source', depthTexture, 0);
        gl.uniform2f(
          uniform(blur, 'direction'),
          i % 2 === 0 ? 1 : 0,
          i % 2 === 0 ? 0 : 1,
        );
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        depthTexture = target.texture;
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(surface);
    gl.bindVertexArray(quadVAOs[1]);
    cameraUniforms(surface);
    bindTexture(surface, 'depthMap', depthTexture, 0);
    bindTexture(surface, 'thicknessMap', targets[3].texture, 1);
    f(surface, 'time', fluid.time);
    f(surface, 'lightPower', s.light);
    f(surface, 'reflectionOn', +s.reflection);
    f(surface, 'causticsOn', +s.caustics);
    f(surface, 'particleView', +s.particles);
    f(surface, 'brushOn', +(!!pointer && !pointer.orbit));
    gl.uniform3fv(uniform(surface, 'brush'), pointer?.world || [0, -0.25, 0]);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    frames++;
    if (now - statTime > 700) {
      onStats({
        fps: Math.round((frames * 1000) / (now - statTime)),
        count: fluid.count,
      });
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
        fluid.pour(pointer.world[0], pointer.world[2], 20);
      else
        fluid.splash(
          pointer.world[0],
          pointer.world[2],
          getSettings().strength,
        );
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
      fluid.splash(0, 0, getSettings().strength);
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
    ripple: () => fluid.splash(0, 0, getSettings().strength),
    shake: () => fluid.shake(),
    pour: () => {
      burst = Math.min(burst + 250, 750);
    },
    drain: () => {
      fluid.drain();
      onStats({ fps: 0, count: fluid.count });
    },
    reset: () => {
      fluid.reset();
      accumulator = 0;
      burst = 0;
      pointer = null;
      yaw = 0.58;
      pitch = 0.49;
      zoom = 8;
      onStats({ fps: 0, count: fluid.count });
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
      destroyTargets();
      gl.deleteBuffer(particleBuffer);
      gl.deleteBuffer(quadBuffer);
      gl.deleteVertexArray(particleVAO);
      quadVAOs.forEach((v) => gl.deleteVertexArray(v));
      programs.forEach((p) => gl.deleteProgram(p));
    },
  };
}
