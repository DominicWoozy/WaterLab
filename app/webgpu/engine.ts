import { WebGPUSimulation, type WebGPUQuality } from './simulation.ts';
import { WebGPUVolume } from './volume.ts';
import { WebGPURenderer } from './renderer.ts';
import type { WaterSettings, WaterStats } from '../water-engine.ts';
import type { GpuFluidAction } from '../gpu-fluid.ts';
export async function createWebGPUWater(
  canvas: HTMLCanvasElement,
  getSettings: () => WaterSettings,
  onStats: (s: WaterStats) => void,
  onError: (s: string) => void,
  device: GPUDevice,
) {
  const resources = await Promise.allSettled([
    WebGPUSimulation.create(device),
    WebGPUVolume.create(device),
    WebGPURenderer.create(device, navigator.gpu.getPreferredCanvasFormat()),
  ]);
  if (resources.some((r) => r.status === 'rejected')) {
    for (const r of resources) if (r.status === 'fulfilled') r.value.destroy();
    throw (
      resources.find((r) => r.status === 'rejected') as PromiseRejectedResult
    ).reason;
  }
  const sim = (resources[0] as PromiseFulfilledResult<WebGPUSimulation>).value;
  const volume = (resources[1] as PromiseFulfilledResult<WebGPUVolume>).value;
  const renderer = (resources[2] as PromiseFulfilledResult<WebGPURenderer>)
    .value;
  const context = canvas.getContext('webgpu');
  if (!context) {
    sim.destroy();
    volume.destroy();
    renderer.destroy();
    throw new Error('无法创建 WebGPU 画布');
  }
  context.configure({
    device,
    format: navigator.gpu.getPreferredCanvasFormat(),
    alphaMode: 'opaque',
  });
  let disposed = false,
    lostContext = false,
    lastUpdate = 0,
    raf = 0,
    frames = 0,
    statTime = performance.now(),
    renderScale = 1;
  let yaw = 0.58,
    pitch = 0.49,
    zoom = 8,
    accumulator = 0,
    pendingPour = 0,
    shakeUntil = 0,
    dirty = true,
    needsReset = true,
    inFlight = 0;
  let resizePending = true;
  let requestedQuality: WebGPUQuality = 50000,
    pourAt = [-0.7, 0],
    splash = [0, 0, 0];
  const actions: GpuFluidAction[] = [];
  const abort = new AbortController();
  const modelReady = renderer.loadModel(abort.signal).catch((e) => {
    if (!disposed) onError(e instanceof Error ? e.message : '鸭子模型加载失败');
  });
  void modelReady;
  const normalize = (a: number[]) => {
    const n = Math.hypot(...a);
    return a.map((v) => v / n);
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
  const resize = () => {
    const r = canvas.getBoundingClientRect(),
      scale = Math.min(
        window.devicePixelRatio,
        1.25,
        (1250 * renderScale) / Math.max(1, r.width),
      );
    const w = Math.max(1, Math.round(r.width * scale)),
      h = Math.max(1, Math.round(r.height * scale));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  };
  // Changing canvas.width/height clears the presented image. Queue all size
  // changes until the start of a frame that will actually submit a new image.
  // In particular, never resize after queue.submit in the FPS feedback block.
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
    const r = canvas.getBoundingClientRect();
    let u = (x - r.left - r.width * 0.5) / r.height;
    if (r.width / r.height > 1.18) u += 0.19;
    const v = -(y - r.top - r.height * 0.5) / r.height,
      c = camera(),
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
  const fail = (e: unknown) => {
    if (disposed) return;
    lostContext = true;
    onError(e instanceof Error ? e.message : 'WebGPU 计算失败，请刷新重试。');
  };
  device.addEventListener('uncapturederror', (e) => fail(e.error));
  void device.lost.then((info) => {
    if (!disposed && info.reason !== 'destroyed')
      fail(new Error(`WebGPU 设备断开：${info.message}`));
  });
  const render = (now: number) => {
    if (disposed || lostContext) return;
    raf = requestAnimationFrame(render);
    if (inFlight >= 2 || document.hidden) {
      lastUpdate = now;
      return;
    }
    if (resizePending) {
      resize();
      resizePending = false;
    }
    const s = getSettings();
    const elapsed = lastUpdate
      ? Math.min(0.05, (now - lastUpdate) / 1000)
      : 1 / 60;
    lastUpdate = now;
    let encoder = device.createCommandEncoder({ label: 'WaterLab physics' });
    try {
      for (const action of actions.splice(0)) {
        if (action.type === 'quality') {
          requestedQuality = action.count;
          needsReset = true;
        }
        if (action.type === 'reset') {
          requestedQuality = sim.quality;
          needsReset = true;
        }
        if (action.type === 'drain') {
          sim.count = Math.max(0, sim.count - (action.amount ?? 500));
          dirty = true;
        }
        if (action.type === 'pour') {
          pendingPour = Math.min(2000, pendingPour + (action.amount ?? 500));
          pourAt = [action.x ?? -0.7, action.z ?? 0];
        }
        if (action.type === 'splash')
          splash = [action.x ?? 0, action.z ?? 0, action.strength ?? 1];
        if (action.type === 'shake') shakeUntil = sim.time + 0.8;
      }
      if (needsReset) {
        sim.reset(encoder, requestedQuality);
        needsReset = false;
        accumulator = 0;
        pendingPour = 0;
        shakeUntil = 0;
        splash = [0, 0, 0];
        dirty = true;
      }
      const budget = elapsed > 1 / 45 ? 1 : 2;
      accumulator = s.paused
        ? 0
        : Math.min(budget / 60, accumulator + elapsed * s.speed);
      for (
        let step = 0;
        step < budget && accumulator + 1e-8 >= 1 / 60;
        step++
      ) {
        const b = pointer && !pointer.orbit ? pointer : null;
        if (b && s.mode === 'pour') {
          pendingPour = Math.min(2000, pendingPour + 14);
          pourAt = [b.world[0], b.world[2]];
        }
        const previousCount = sim.count,
          added = Math.min(18, pendingPour, 50000 - sim.count);
        sim.count += added;
        pendingPour -= added;
        sim.step(
          encoder,
          {
            previousCount,
            surfaceTension: s.surfaceTension,
            forces: {
              gravity: s.gravity,
              agitation: s.agitation,
              viscosity: s.viscosity,
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
            pourAt,
            splash,
            shake:
              sim.time < shakeUntil
                ? Math.sin((shakeUntil - sim.time) * 16) * 22
                : 0,
          },
          step,
        );
        splash = [0, 0, 0];
        accumulator -= 1 / 60;
        dirty = true;
      }
      // Separate stage command buffers avoid oversized Metal encoder workloads;
      // submit them together in order, with no CPU wait or particle readback.
      const commands = [encoder.finish()];
      if (volume.detailsEnabled !== s.details) dirty = true;
      if (dirty && !s.particles) {
        encoder = device.createCommandEncoder({ label: 'WaterLab density' });
        volume.encode(encoder, sim, s.details);
        commands.push(encoder.finish());
        dirty = false;
      }
      encoder = device.createCommandEncoder({ label: 'WaterLab render' });
      renderer.encode(
        encoder,
        context.getCurrentTexture().createView(),
        sim,
        volume,
        camera(),
        canvas.width,
        canvas.height,
        {
          light: s.light,
          reflection: s.reflection,
          caustics: s.caustics,
          particles: s.particles,
          brush: pointer && !pointer.orbit ? [...pointer.world, 1] : undefined,
        },
      );
      device.queue.submit([...commands, encoder.finish()]);
      inFlight++;
      void device.queue.onSubmittedWorkDone().then(() => {
        inFlight--;
      }, fail);
      frames++;
      if (now - statTime > 1000) {
        const fps = (frames * 1000) / (now - statTime);
        onStats({
          fps: Math.round(fps),
          count: sim.count,
          quality: sim.quality,
          capacity: 50000,
          backend: 'WebGPU',
        });
        const next =
          fps < 35
            ? Math.max(0.65, renderScale * 0.9)
            : fps > 56
              ? Math.min(1.12, renderScale * 1.03)
              : renderScale;
        if (Math.abs(next - renderScale) > 0.015) {
          renderScale = next;
          resizePending = true;
        }
        frames = 0;
        statTime = now;
      }
    } catch (e) {
      fail(e);
    }
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

  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', cancel);
  canvas.addEventListener('wheel', wheel, { passive: false });
  canvas.addEventListener('keydown', key);
  canvas.addEventListener('contextmenu', contextMenu);
  onStats({
    fps: 0,
    count: 50000,
    quality: 50000,
    capacity: 50000,
    backend: 'WebGPU',
  });
  raf = requestAnimationFrame(render);
  return {
    setQuality: (count: WebGPUQuality) => {
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
      abort.abort();
      cancelAnimationFrame(raf);
      observer.disconnect();
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', up);
      canvas.removeEventListener('pointercancel', cancel);
      canvas.removeEventListener('wheel', wheel);
      canvas.removeEventListener('keydown', key);
      canvas.removeEventListener('contextmenu', contextMenu);
      renderer.destroy();
      volume.destroy();
      sim.destroy();
      context.unconfigure();
      device.destroy();
    },
  };
}
