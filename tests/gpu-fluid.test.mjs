import test from 'node:test';
import assert from 'node:assert/strict';
import { GpuFluid } from '../app/gpu-fluid.ts';
import * as shaders from '../app/gpu-fluid-shaders.ts';

// Command-contract test; arithmetic is tested separately on a real GPU.
function recordingGL(floatSupport = true) {
  let serial = 1,
    current,
    framebuffer;
  const draws = [],
    deleted = [];
  const counters = { scalarWrites: 0, programBinds: 0, viewportWrites: 0 };
  let viewport;
  const gl = {
    draws,
    deleted,
    counters,
    getExtension: () => (floatSupport ? {} : null),
    createTexture: () => ({ id: serial++ }),
    createFramebuffer: () => ({ id: serial++ }),
    createVertexArray: () => ({ id: serial++ }),
    createShader: () => ({}),
    createProgram: () => ({ sources: [], values: {} }),
    shaderSource: (s, source) => {
      s.source = source;
    },
    compileShader() {},
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    attachShader: (p, s) => {
      p.sources.push(s.source);
    },
    deleteShader() {},
    linkProgram() {},
    getProgramParameter: () => true,
    getProgramInfoLog: () => '',
    useProgram: (p) => {
      current = p;
      counters.programBinds++;
    },
    getUniformLocation: (p, name) => ({ p, name }),
    bindFramebuffer: (_, f) => {
      framebuffer = f;
    },
    framebufferTexture2D: (_a, _b, _c, tex) => {
      framebuffer.texture = tex;
    },
    checkFramebufferStatus() {
      return gl.FRAMEBUFFER_COMPLETE;
    },
    drawBuffers() {},
    bindTexture() {},
    texStorage2D() {},
    texParameteri() {},
    clearColor() {},
    clear() {},
    activeTexture() {},
    viewport(...value) {
      viewport = value;
      counters.viewportWrites++;
    },
    bindVertexArray() {},
    disable() {},
    enable() {},
    blendEquation() {},
    blendFunc() {},
    deleteTexture: (t) => deleted.push(t),
    deleteFramebuffer: (f) => deleted.push(f),
    deleteVertexArray: (v) => deleted.push(v),
    deleteProgram: (p) => deleted.push(p),
  };
  for (const name of [
    'uniform1i',
    'uniform1f',
    'uniform2fv',
    'uniform3fv',
    'uniform4fv',
  ])
    gl[name] = (loc, value) => {
      loc.p.values[loc.name] = value;
      if (name === 'uniform1i' || name === 'uniform1f') counters.scalarWrites++;
    };
  const record = (instances) => {
    const name = Object.entries(shaders).find(
      ([key, source]) =>
        key.endsWith('Fragment') && source === current.sources[1],
    )?.[0];
    draws.push({ name, instances, values: { ...current.values }, viewport });
  };
  gl.drawArrays = () => record(1);
  gl.drawArraysInstanced = (_a, _b, _c, instances) => record(instances);
  // Deliberately provide no CPU readback, CPU position upload, or Worker methods.
  for (const name of [
    'TEXTURE_2D',
    'RGBA32F',
    'R16F',
    'LINEAR',
    'NEAREST',
    'TEXTURE_MIN_FILTER',
    'TEXTURE_MAG_FILTER',
    'TEXTURE_WRAP_S',
    'TEXTURE_WRAP_T',
    'CLAMP_TO_EDGE',
    'FRAMEBUFFER',
    'COLOR_ATTACHMENT0',
    'FRAMEBUFFER_COMPLETE',
    'COLOR_BUFFER_BIT',
    'VERTEX_SHADER',
    'FRAGMENT_SHADER',
    'COMPILE_STATUS',
    'LINK_STATUS',
    'DEPTH_TEST',
    'BLEND',
    'TEXTURE0',
    'FUNC_ADD',
    'ONE',
    'TRIANGLES',
  ])
    gl[name] = serial++;
  return gl;
}
const job = (overrides = {}) => ({
  elapsed: 1 / 60,
  speed: 1,
  paused: false,
  particles: false,
  forces: { gravity: 9.8, agitation: 0, viscosity: 0.025 },
  actions: [],
  ...overrides,
});

test('GPU update dispatches full physics, volume and bounds without state readback/upload', () => {
  const gl = recordingGL(),
    fluid = new GpuFluid(gl);
  gl.draws.length = 0;
  fluid.update(job());
  assert.equal(fluid.count, 15000);
  assert.equal(
    gl.draws.filter((d) =>
      ['sortFragment', 'sortMergeFragment'].includes(d.name),
    ).length,
    81,
  );
  assert.equal(gl.draws.filter((d) => d.name === 'correctFragment').length, 3);
  assert.equal(gl.draws.filter((d) => d.name === 'boundsFragment').length, 8);
  assert.equal(
    gl.draws.filter((d) => d.name === 'divergenceFactorFragment').length,
    1,
  );
  assert.equal(
    gl.draws.filter((d) => d.name === 'divergenceProjectFragment').length,
    2,
  );
  assert.equal(
    gl.draws.filter((d) => d.name === 'surfaceFilterFragment').length,
    3,
  );
  assert.equal(
    gl.draws.find((d) => d.name === 'volumeFragment').instances,
    300000,
  );
  assert.equal(
    gl.draws.find((d) => d.name === 'predictFragment').values.dt,
    1 / 60,
  );
  assert.ok(fluid.positions && fluid.volume && fluid.volumeBounds);
  fluid.destroy();
  assert.ok(gl.deleted.length > 40);
});

test('pause is inert; debug skips reconstruction, and switching back rebuilds once', () => {
  const gl = recordingGL(),
    fluid = new GpuFluid(gl);
  fluid.update(job());
  gl.draws.length = 0;
  fluid.update(job({ paused: true, elapsed: 20 }));
  assert.equal(gl.draws.length, 0);
  assert.equal(fluid.time, 1 / 60);
  fluid.update(job({ particles: true }));
  assert.ok(!gl.draws.some((d) => d.name === 'volumeFragment'));
  gl.draws.length = 0;
  fluid.update(job({ paused: true }));
  assert.equal(gl.draws.filter((d) => d.name === 'volumeFragment').length, 1);
});

test('GPU actions retain count limits, staged injection, reset, and fixed timestep budget', () => {
  const gl = recordingGL(),
    fluid = new GpuFluid(gl);
  fluid.update(job({ actions: [{ type: 'pour', amount: 2000 }] }));
  assert.equal(fluid.count, 15018);
  assert.equal(
    gl.draws.find((d) => d.name === 'predictFragment').values.previousCount,
    15000,
  );
  for (let i = 0; i < 112; i++) {
    gl.draws.length = 0;
    fluid.update(job({ particles: true }));
  }
  assert.equal(fluid.count, 17000);
  fluid.update(
    job({ actions: [{ type: 'drain', amount: 30000 }], paused: true }),
  );
  assert.equal(fluid.count, 0);
  assert.equal(
    gl.draws.findLast((d) => d.name === 'volumeFragment').instances,
    0,
  );
  fluid.update(job({ actions: [{ type: 'reset' }], paused: true }));
  assert.equal(fluid.count, 15000);
  assert.equal(fluid.time, 0);
  gl.draws.length = 0;
  fluid.update(job({ elapsed: 10, speed: 2 }));
  assert.equal(gl.draws.filter((d) => d.name === 'predictFragment').length, 1);
});

test('unsupported GPU reports a capability error instead of silently reverting to CPU', () => {
  assert.throws(() => new GpuFluid(recordingGL(false)), /浮点渲染支持/);
});

test('30,000 quality uses full sort range; switching back resets scale and restores smaller sort', () => {
  const gl = recordingGL(),
    fluid = new GpuFluid(gl);
  fluid.update(
    job({ actions: [{ type: 'quality', count: 30000 }], paused: true }),
  );
  assert.equal(fluid.count, 30000);
  assert.equal(
    gl.draws.filter((d) =>
      ['sortFragment', 'sortMergeFragment'].includes(d.name),
    ).length,
    94,
  );
  const shape = gl.draws.find((d) => d.name === 'geometryFragment');
  assert.ok(shape);
  assert.ok(Math.abs(shape.values.particleScale - Math.cbrt(1 / 3)) < 1e-9);
  gl.draws.length = 0;
  fluid.update(
    job({ actions: [{ type: 'quality', count: 15000 }], paused: true }),
  );
  assert.equal(fluid.count, 15000);
  assert.equal(
    gl.draws.filter((d) =>
      ['sortFragment', 'sortMergeFragment'].includes(d.name),
    ).length,
    81,
  );
});

test('spatial reorder survives odd iterations, pause/drain, and quality changes without state aliasing', () => {
  const gl = recordingGL(),
    fluid = new GpuFluid(gl);
  fluid.update(
    job({ actions: [{ type: 'quality', count: 30000 }], paused: true }),
  );
  for (let frame = 0; frame < 40; frame++) {
    gl.draws.length = 0;
    fluid.update(job());
    assert.equal(
      gl.draws.filter((d) => d.name === 'reorderFragment').length,
      1,
    );
    assert.equal(
      gl.draws.filter((d) => d.name === 'correctFragment').length,
      3,
    );
  }
  fluid.update(
    job({ actions: [{ type: 'drain', amount: 500 }], paused: true }),
  );
  fluid.update(job({ actions: [{ type: 'pour', amount: 500 }] }));
  assert.equal(fluid.count, 29518);
  fluid.update(
    job({ actions: [{ type: 'quality', count: 15000 }], paused: true }),
  );
  for (let frame = 0; frame < 10; frame++) fluid.update(job());
  assert.equal(fluid.count, 15000);
});

test('slow frames do not amplify GPU load; fast frames retain speed control', () => {
  const gl = recordingGL(),
    fluid = new GpuFluid(gl);
  for (let frame = 0; frame < 120; frame++) {
    gl.draws.length = 0;
    fluid.update(job({ elapsed: 1 / 30, particles: true }));
    const steps = gl.draws.filter((d) => d.name === 'predictFragment');
    assert.equal(steps.length, 1);
    assert.equal(steps[0].values.dt, 1 / 60);
  }
  assert.ok(Math.abs(fluid.time - 2) < 1e-8);
  assert.equal(fluid.count, 15000);
  gl.draws.length = 0;
  fluid.update(job({ elapsed: 1 / 60, speed: 2, particles: true }));
  assert.equal(gl.draws.filter((d) => d.name === 'predictFragment').length, 2);
  gl.draws.length = 0;
  fluid.update(job({ paused: true, elapsed: 5, particles: true }));
  fluid.update(job({ elapsed: 1 / 60, particles: true }));
  assert.equal(gl.draws.filter((d) => d.name === 'predictFragment').length, 1);
});

test('compute caches redundant GL submissions and restores state after scene rendering', () => {
  const gl = recordingGL(),
    fluid = new GpuFluid(gl);
  fluid.update(
    job({ actions: [{ type: 'quality', count: 30000 }], particles: true }),
  );
  gl.useProgram({ sources: [], values: {} });
  gl.viewport(0, 0, 1234, 567);
  for (const key of Object.keys(gl.counters)) gl.counters[key] = 0;
  gl.draws.length = 0;
  fluid.update(job({ particles: true }));
  assert.equal(gl.draws[0].name, 'duckPredictFragment');
  assert.deepEqual(gl.draws[0].viewport, [0, 0, 4, 1]);
  assert.deepEqual(
    gl.draws.find((d) => d.name === 'predictFragment').viewport,
    [0, 0, 256, 118],
  );
  assert.ok(
    gl.draws
      .filter((d) => d.name.startsWith('sort'))
      .every((d) => d.viewport[3] === 128),
  );
  assert.ok(gl.counters.scalarWrites < 180, JSON.stringify(gl.counters));
  assert.ok(gl.counters.programBinds < 50, JSON.stringify(gl.counters));
  assert.ok(gl.counters.viewportWrites < 24, JSON.stringify(gl.counters));
});
