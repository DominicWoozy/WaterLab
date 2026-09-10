import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import ts from 'typescript';

// Exercise the real animation loop with a canvas that loses its image whenever
// either backing dimension changes. GPU arithmetic has separate native tests.
const source = await readFile(
  new URL('../app/webgpu/engine.ts', import.meta.url),
  'utf8',
);
const code = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const glSource = await readFile(
  new URL('../app/water-engine.ts', import.meta.url),
  'utf8',
);
const glCode = ts.transpileModule(glSource, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
async function harness(backend = 'webgpu') {
  let frame,
    observer,
    width = 300,
    height = 150,
    now = 0,
    hold = false;
  let rect = { width: 1250, height: 800, left: 0, top: 0 };
  const trace = [],
    canvasEvents = [],
    exports = {},
    document = { hidden: false };
  let presented = 'blank';
  const context = {
    configure() {},
    unconfigure() {},
    getCurrentTexture: () => ({ createView: () => ({}) }),
  };
  const gl = new Proxy(
    {},
    {
      get(_target, key) {
        if (key === 'drawArrays')
          return () => {
            trace.push('render', 'submit');
            presented = 'water';
          };
        if (key.startsWith('create') || key === 'getUniformLocation')
          return () => ({});
        if (key === 'getShaderParameter' || key === 'getProgramParameter')
          return () => true;
        if (key === 'getAttribLocation') return () => 0;
        return /^[A-Z_0-9]+$/.test(key) ? 1 : () => {};
      },
    },
  );
  const canvas = {
    get width() {
      return width;
    },
    set width(v) {
      width = v;
      presented = 'blank';
      trace.push('resize');
    },
    get height() {
      return height;
    },
    set height(v) {
      height = v;
      presented = 'blank';
      trace.push('resize');
    },
    getContext: () => (backend === 'webgpu' ? context : gl),
    getBoundingClientRect: () => rect,
    addEventListener: (...v) => canvasEvents.push(v),
    removeEventListener() {},
    setPointerCapture() {},
  };
  const sim = {
    quality: 50000,
    count: 50000,
    time: 0,
    reset(_encoder, quality = this.quality) {
      this.count = this.quality = quality;
      this.time = 0;
    },
    step() {
      trace.push('physics');
      this.time += 1 / 60;
    },
    destroy() {},
  };
  const volume = {
    detailsEnabled: false,
    encode(_e, _s, details, reuse) {
      trace.push(reuse ? 'density-reuse' : 'density-rebuild');
      this.detailsEnabled = details;
    },
    destroy() {},
  };
  const renderer = {
    loadModel: async () => {},
    encode() {
      trace.push('render');
    },
    destroy() {},
  };
  const factories = {
    './common.ts': { CAPACITY: 100000 },
    './duck-model': { loadDuckModel: () => ({ ready: true, destroy() {} }) },
    './gpu-volume-config': { GPU_VOLUME_SIZE: [128, 160, 96] },
    './fluid-volume': {
      VOLUME_MIN: [-2, -1, -2],
      VOLUME_MAX: [2, 4, 2],
      SURFACE_DENSITY: 1.15,
      ABSORPTION: [1, 0.2, 0.06],
    },
    './gpu-fluid': {
      GpuFluid: class {
        count = 15000;
        quality = 15000;
        time = 0;
        update(job) {
          for (const action of job.actions ?? [])
            if (action.type === 'quality') {
              trace.push(`quality-${action.count}`);
              this.count = this.quality = action.count;
            }
          this.time += 1 / 60;
        }
        destroy() {}
      },
    },
    './water-shaders': {},
    './simulation.ts': { WebGPUSimulation: { create: async () => sim } },
    './volume.ts': { WebGPUVolume: { create: async () => volume } },
    './renderer.ts': { WebGPURenderer: { create: async () => renderer } },
  };
  const settings = {
    paused: false,
    speed: 1,
    mode: 'stir',
    details: true,
    surfaceTension: true,
    gravity: 9.8,
    agitation: 0,
    viscosity: 0.025,
    light: 1.3,
    reflection: true,
    caustics: true,
    particles: false,
  };
  const device = {
    addEventListener() {},
    lost: new Promise(() => {}),
    destroy() {},
    createCommandEncoder: () => ({ finish: () => ({}) }),
    queue: {
      submit() {
        trace.push('submit');
        presented = 'water';
      },
      onSubmittedWorkDone: () =>
        hold ? new Promise(() => {}) : Promise.resolve(),
    },
  };
  vm.runInNewContext(backend === 'webgpu' ? code : glCode, {
    exports,
    require: (name) => factories[name],
    window: { devicePixelRatio: 1 },
    document,
    navigator: { gpu: { getPreferredCanvasFormat: () => 'rgba8unorm' } },
    performance: { now: () => now },
    AbortController,
    ResizeObserver: class {
      constructor(f) {
        observer = f;
      }
      observe() {}
      disconnect() {}
    },
    requestAnimationFrame: (f) => ((frame = f), 1),
    cancelAnimationFrame() {},
    console,
  });
  const errors = [];
  const createEngine =
    backend === 'webgpu' ? exports.createWebGPUWater : exports.createWebGLWater;
  const engine = await createEngine(
    canvas,
    () => settings,
    () => {},
    (e) => errors.push(e),
    device,
  );
  return {
    engine,
    sim,
    document,
    settings,
    trace,
    errors,
    get presented() {
      return presented;
    },
    get width() {
      return width;
    },
    resize(w, h) {
      rect = { ...rect, width: w, height: h };
      observer();
    },
    hold() {
      hold = true;
    },
    async frame(t) {
      now = t;
      trace.length = 0;
      frame(t);
      await Promise.resolve();
      await Promise.resolve();
      return [...trace];
    },
  };
}

test('adaptive resolution never clears an already submitted frame', async () => {
  const h = await harness();
  for (let i = 1; i <= 45; i++) {
    const trace = await h.frame(i * 100); // deliberately slow: exercise repeated downscaling
    assert.equal(h.presented, 'water', `frame ${i} ends with a valid picture`);
    assert.equal(trace.at(-1), 'submit');
    if (trace.includes('resize'))
      assert.ok(trace.lastIndexOf('resize') < trace.indexOf('render'));
  }
  assert.ok(h.width < 1250, 'adaptive resolution still reduces rendering cost');
  assert.deepEqual(h.errors, []);
  h.engine.destroy();
});

test('observer, paused mode and hidden frames do not clear the displayed image', async () => {
  const h = await harness();
  await h.frame(16);
  h.settings.paused = true;
  h.resize(1000, 700);
  assert.equal(h.presented, 'water');
  h.document.hidden = true;
  assert.deepEqual(await h.frame(32), []);
  assert.equal(h.presented, 'water');
  h.document.hidden = false;
  const trace = await h.frame(48);
  assert.ok(trace.includes('resize'));
  assert.equal(trace.at(-1), 'submit');
  assert.equal(h.presented, 'water');
  h.engine.destroy();
});

test('GPU backpressure defers resize together with the skipped frame', async () => {
  const h = await harness();
  await h.frame(16);
  h.hold();
  await h.frame(32);
  await h.frame(48);
  const width = h.width;
  h.resize(800, 500);
  assert.deepEqual(await h.frame(64), []);
  assert.equal(h.width, width);
  assert.equal(h.presented, 'water');
  h.engine.destroy();
});

for (const dt of [16, 100])
  test(`WebGL2 resizing never clears a completed frame (${dt} ms cadence)`, async () => {
    const h = await harness('webgl');
    for (let i = 1; i <= 160; i++) {
      if (i === 20) {
        h.resize(1200, 760);
        assert.equal(h.presented, 'water');
      }
      const trace = await h.frame(i * dt);
      assert.equal(h.presented, 'water');
      assert.equal(trace.at(-1), 'submit');
      if (trace.includes('resize'))
        assert.ok(trace.lastIndexOf('resize') < trace.indexOf('render'));
    }
    assert.deepEqual(h.errors, []);
    h.engine.destroy();
  });

test('reconstruction reuses neighbors only after a completed physical tick', async () => {
  const h = await harness();
  h.settings.paused = true;
  assert.ok(
    (await h.frame(16)).includes('density-rebuild'),
    'paused initial reset',
  );
  h.settings.paused = false;
  const active = await h.frame(36);
  assert.ok(active.indexOf('physics') < active.indexOf('density-reuse'));
  assert.ok(active.includes('density-reuse'));
  h.settings.paused = true;
  h.engine.drain();
  assert.ok(
    (await h.frame(52)).includes('density-rebuild'),
    'drain invalidates particle ranges',
  );
  h.engine.reset();
  assert.ok(
    (await h.frame(68)).includes('density-rebuild'),
    'reset invalidates neighbor lists',
  );
  h.settings.details = false;
  assert.ok(
    (await h.frame(84)).includes('density-rebuild'),
    'paused detail toggle stays conservative',
  );
  h.settings.paused = false;
  h.settings.particles = true;
  assert.ok(!(await h.frame(104)).includes('density-reuse'));
  h.settings.paused = true;
  h.settings.particles = false;
  assert.ok(
    (await h.frame(120)).includes('density-rebuild'),
    'leaving debug while paused',
  );
  h.settings.paused = false;
  h.settings.speed = 2;
  const doubleTick = await h.frame(140);
  assert.equal(doubleTick.filter((x) => x === 'physics').length, 2);
  assert.ok(
    doubleTick.includes('density-reuse'),
    'two ticks use the final tick cache',
  );
  assert.deepEqual(h.errors, []);
  h.engine.destroy();
});

// High presets must survive reset and the engine's pending-action queue.
test('70k and 100k quality changes reach the simulation and can switch back', async () => {
  const h = await harness();
  await h.frame(16);
  let now = 32;
  for (const count of [70000, 100000, 15000, 100000, 50000]) {
    h.engine.setQuality(count);
    await h.frame(now);
    now += 32;
    assert.equal(h.sim.quality, count);
    assert.equal(h.sim.count, count);
  }
  assert.deepEqual(h.errors, []);
  h.engine.destroy();
});

test('WebGL2 maps unsupported high particle presets to 30k', async () => {
  const h = await harness('webgl');
  await h.frame(16);
  let now = 32;
  for (const count of [50000, 70000, 100000]) {
    h.engine.setQuality(count);
    assert.ok((await h.frame(now)).includes('quality-30000'));
    now += 32;
  }
  assert.deepEqual(h.errors, []);
  h.engine.destroy();
});
