import test from 'node:test';
import assert from 'node:assert/strict';
import {
  FluidVolume,
  SURFACE_DENSITY,
  transmittance,
} from '../app/fluid-volume.ts';
import { ParticleFluid } from '../app/fluid-simulation.ts';
function slab(layers) {
  const p = [];
  for (let y = 0; y < layers; y++)
    for (let z = -16; z <= 16; z++)
      for (let x = -24; x <= 24; x++)
        p.push(x * 0.07, -0.85 + y * 0.09, z * 0.07);
  const volume = new FluidVolume();
  volume.rebuild(new Float32Array(p), p.length / 3);
  return volume;
}
function surfaceHeight(volume, x, z) {
  let top = -1;
  for (let y = 1; y > -0.9; y -= 0.004) {
    if (volume.sample(x, y, z) > SURFACE_DENSITY) {
      top = y;
      break;
    }
  }
  return top;
}
test('a flat particle slab reconstructs one smooth surface, without per-particle spherical bumps', () => {
  const volume = slab(6),
    heights = [];
  for (let z = -0.7; z <= 0.7; z += 0.061)
    for (let x = -1; x <= 1; x += 0.059)
      heights.push(surfaceHeight(volume, x, z));
  const min = Math.min(...heights),
    max = Math.max(...heights);
  console.log(
    `Planar surface variation: ${((max - min) * 1000).toFixed(2)} mm`,
  );
  assert.ok(max - min < 0.015);
  assert.ok(min > -0.5);
});
test('optical path distinguishes shallow and deep reconstructed water', () => {
  const shallow = slab(3),
    deep = slab(10);
  const a = shallow.thickness([0, 1, 0], [0, -1, 0], 2),
    b = deep.thickness([0, 1, 0], [0, -1, 0], 2);
  console.log(
    `Optical path: shallow ${a.toFixed(3)} m, deep ${b.toFixed(3)} m`,
  );
  assert.ok(a > 0.08);
  assert.ok(b > a * 1.7);
  const thin = transmittance(a),
    thick = transmittance(b);
  assert.ok(thick[0] < thin[0]);
  assert.ok(thick[2] / thick[0] > thin[2] / thin[0]);
});
test('absorption preserves clear thin water and never saturates thickness into one fixed colour', () => {
  assert.deepEqual(transmittance(0), [1, 1, 1]);
  assert.ok(transmittance(0.01).every((x) => x > 0.98));
  assert.ok(transmittance(2)[0] < transmittance(1)[0]);
  assert.ok(transmittance(3)[0] < transmittance(2)[0]);
});
test('empty space has zero density and optical depth', () => {
  const volume = new FluidVolume();
  assert.equal(volume.sample(0, 0, 0), 0);
  assert.equal(volume.thickness([0, 1, 0], [0, -1, 0], 2), 0);
});
test('volume follows moved particles in world space', () => {
  const p = new Float32Array([0, 0, 0]),
    volume = new FluidVolume();
  volume.rebuild(p, 1, new Float32Array([0]));
  assert.ok(volume.sample(0, 0, 0) > SURFACE_DENSITY);
  p[0] = 1;
  volume.rebuild(p, 1, new Float32Array([0]));
  assert.equal(volume.sample(0, 0, 0), 0);
  assert.ok(volume.sample(1, 0, 0) > SURFACE_DENSITY);
});
test('settled physical water reconstructs a level free surface', () => {
  const fluid = new ParticleFluid();
  for (let i = 0; i < 480; i++)
    fluid.step(1 / 120, { gravity: 9.8, viscosity: 0.025, agitation: 0 });
  const volume = new FluidVolume(),
    begin = performance.now();
  volume.rebuild(fluid.positions, fluid.count, fluid.densities);
  console.log(`Volume rebuild: ${(performance.now() - begin).toFixed(2)} ms`);
  const heights = [];
  for (let z = -0.6; z <= 0.6; z += 0.12)
    for (let x = -0.9; x <= 0.9; x += 0.12)
      heights.push(surfaceHeight(volume, x, z));
  const mean = heights.reduce((a, b) => a + b) / heights.length;
  const deviation = Math.sqrt(
    heights.reduce((sum, y) => sum + (y - mean) ** 2, 0) / heights.length,
  );
  console.log(`Settled surface deviation: ${(deviation * 1000).toFixed(2)} mm`);
  assert.ok(deviation < 0.045);
});
