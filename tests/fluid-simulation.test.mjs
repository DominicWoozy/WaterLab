import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ParticleFluid,
  CAPACITY,
  FLOOR,
  HALF_X,
  HALF_Z,
} from '../app/fluid-simulation.ts';
const calm = { gravity: 9.8, viscosity: 0.12, agitation: 0 };
const average = (f, axis) => {
  let n = 0;
  for (let i = axis; i < f.count * 3; i += 3) n += f.positions[i];
  return n / f.count;
};
const kinetic = (f) => {
  let n = 0;
  for (let i = 0; i < f.count * 3; i++) n += f.velocities[i] ** 2;
  return n / Math.max(1, f.count);
};
test('free particles accelerate downwards with gravity', () => {
  const f = new ParticleFluid(1);
  f.positions.set([0, 1, 0]);
  f.velocities.fill(0);
  for (let i = 0; i < 12; i++) f.step(1 / 120, calm);
  assert.ok(f.positions[1] < 0.98);
  assert.ok(f.velocities[1] < -0.8);
});
test('zero gravity preserves a stationary isolated particle', () => {
  const f = new ParticleFluid(1);
  f.positions.set([0, 1, 0]);
  f.velocities.fill(0);
  for (let i = 0; i < 30; i++) f.step(1 / 120, { ...calm, gravity: 0 });
  assert.equal(f.positions[1], 1);
});
test('released water settles, preserves particle count, and stays finite inside the basin', () => {
  const f = new ParticleFluid(1700);
  const start = average(f, 1);
  const started = performance.now();
  for (let step = 0; step < 360; step++) {
    if (step === 150) f.splash();
    if (step === 210) f.shake();
    f.step(1 / 120, calm);
  }
  assert.equal(f.count, 1700);
  assert.ok(average(f, 1) < start + 0.1);
  for (let i = 0; i < f.count * 3; i += 3) {
    assert.ok(
      Number.isFinite(f.positions[i]) &&
        Number.isFinite(f.positions[i + 1]) &&
        Number.isFinite(f.positions[i + 2]),
    );
    assert.ok(Math.abs(f.positions[i]) <= HALF_X + 0.0001);
    assert.ok(Math.abs(f.positions[i + 2]) <= HALF_Z + 0.0001);
    assert.ok(f.positions[i + 1] >= FLOOR + 0.0329);
    assert.ok(f.positions[i + 1] <= 3.801);
  }
  assert.ok(Number.isFinite(kinetic(f)));
  console.log(
    `1700 particles: ${((performance.now() - started) / 360).toFixed(2)} ms/substep; mean height ${average(f, 1).toFixed(3)}; energy ${kinetic(f).toFixed(3)}`,
  );
});
test('stirring changes momentum and can lift nearby particles', () => {
  const f = new ParticleFluid(200);
  const before = kinetic(f);
  f.stir(-0.5, -0.4, 0, 2, 1, 1, 1 / 60);
  assert.ok(kinetic(f) > before);
  assert.ok(f.velocities.some((v, i) => i % 3 === 1 && v > 0));
});
test('pour and drain respect capacity and exact count', () => {
  const f = new ParticleFluid(100);
  assert.equal(f.pour(0, 0, 50), 50);
  assert.equal(f.count, 150);
  f.pour(0, 0, CAPACITY);
  assert.equal(f.count, CAPACITY);
  assert.equal(f.pour(), 0);
  f.drain(250);
  assert.equal(f.count, CAPACITY - 250);
  f.drain(CAPACITY);
  assert.equal(f.count, 0);
  f.step(1 / 120, calm);
  assert.equal(f.count, 0);
});
test('reset restores the same deterministic initial state', () => {
  const f = new ParticleFluid(200),
    start = f.positions.slice(0, 600);
  f.splash();
  for (let i = 0; i < 12; i++) f.step(1 / 120, calm);
  f.reset(200);
  assert.deepEqual(f.positions.slice(0, 600), start);
});
test('rejects unstable timesteps', () => {
  const f = new ParticleFluid(1);
  assert.throws(() => f.step(0.5, calm));
  assert.throws(() => f.step(0, calm));
});
test('capacity fluid remains stable under strong stirring, high viscosity and changing gravity', () => {
  const f = new ParticleFluid(CAPACITY);
  for (let i = 0; i < 360; i++) {
    if (i % 25 === 0) f.splash(0, 0, 2.5);
    f.stir(Math.sin(i * 0.03), 0, Math.cos(i * 0.03), 6, -6, 2.5, 1 / 120);
    f.step(1 / 120, {
      gravity: i < 180 ? 0 : 14,
      viscosity: 1,
      agitation: 1.4,
    });
  }
  assert.equal(f.count, CAPACITY);
  assert.ok(f.positions.subarray(0, f.count * 3).every(Number.isFinite));
  assert.ok(kinetic(f) < 200);
});
test('under-dense particles do not attract into a jelly-like clump', () => {
  const f = new ParticleFluid(2);
  f.positions.set([-0.06, 1, 0, 0.06, 1, 0]);
  f.velocities.fill(0);
  for (let i = 0; i < 60; i++)
    f.step(1 / 120, { gravity: 0, viscosity: 0, agitation: 0 });
  assert.ok(Math.abs(f.positions[3] - f.positions[0] - 0.12) < 0.00001);
});
