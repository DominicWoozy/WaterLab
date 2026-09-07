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
  const gl = {
    draws,
    deleted,
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
    bindTexture() {},
    texStorage2D() {},
    texParameteri() {},
    clearColor() {},
    clear() {},
    activeTexture() {},
    viewport() {},
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
    };
  const record = (instances) => {
    const name = Object.entries(shaders).find(
      ([key, source]) =>
        key.endsWith('Fragment') && source === current.sources[1],
    )?.[0];
    draws.push({ name, instances, values: { ...current.values } });
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
  assert.equal(fluid.count, 10000);
  assert.equal(gl.draws.filter((d) => d.name === 'sortFragment').length, 105);
  assert.equal(gl.draws.filter((d) => d.name === 'correctFragment').length, 2);
  assert.equal(gl.draws.filter((d) => d.name === 'boundsFragment').length, 7);
  assert.equal(
    gl.draws.find((d) => d.name === 'volumeFragment').instances,
    120000,
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
  assert.equal(fluid.count, 10018);
  assert.equal(
    gl.draws.find((d) => d.name === 'predictFragment').values.previousCount,
    10000,
  );
  for (let i = 0; i < 112; i++) {
    gl.draws.length = 0;
    fluid.update(job({ particles: true }));
  }
  assert.equal(fluid.count, 12000);
  fluid.update(
    job({ actions: [{ type: 'drain', amount: 12000 }], paused: true }),
  );
  assert.equal(fluid.count, 0);
  assert.equal(gl.draws.at(-1).instances, 0);
  fluid.update(job({ actions: [{ type: 'reset' }], paused: true }));
  assert.equal(fluid.count, 10000);
  assert.equal(fluid.time, 0);
  gl.draws.length = 0;
  fluid.update(job({ elapsed: 10, speed: 2 }));
  assert.equal(gl.draws.filter((d) => d.name === 'predictFragment').length, 3);
});

test('unsupported GPU reports a capability error instead of silently reverting to CPU', () => {
  assert.throws(() => new GpuFluid(recordingGL(false)), /浮点渲染支持/);
});
