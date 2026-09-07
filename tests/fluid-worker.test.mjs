import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import {
  DEFAULT_COUNT,
  CAPACITY,
  ParticleFluid,
  KERNEL_RADIUS,
} from '../app/fluid-simulation.ts';

test('default contains 10,000 independently simulated particles', () => {
  const fluid = new ParticleFluid();
  assert.equal(DEFAULT_COUNT, 10000);
  assert.equal(fluid.count, 10000);
  assert.equal(
    new Set(
      Array.from({ length: fluid.count }, (_, i) =>
        fluid.positions.slice(i * 3, i * 3 + 3).join(','),
      ),
    ).size,
    10000,
  );
});
test('sorted neighbor search matches brute force without duplicate pairs', () => {
  const f = new ParticleFluid(400);
  let seed = 711;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 0; i < 400; i++)
    f.positions.set(
      [(random() - 0.5) * 3.5, -0.9 + random() * 2.8, (random() - 0.5) * 2.5],
      i * 3,
    );
  f.buildPairs();
  const actual = new Set();
  for (let a = 0; a < f.pairCount; a++) {
    const i = f.pairI[a],
      j = f.pairJ[a];
    actual.add(`${Math.min(i, j)}:${Math.max(i, j)}`);
  }
  const expected = new Set();
  for (let i = 0; i < 400; i++)
    for (let j = i + 1; j < 400; j++) {
      let r2 = 0;
      for (let axis = 0; axis < 3; axis++)
        r2 += (f.positions[i * 3 + axis] - f.positions[j * 3 + axis]) ** 2;
      if (r2 < KERNEL_RADIUS * KERNEL_RADIUS) expected.add(`${i}:${j}`);
    }
  assert.equal(actual.size, f.pairCount);
  assert.deepEqual(actual, expected);
});
test(
  'real worker computes, transfers and recycles snapshots, and preserves pause/reset',
  { timeout: 15000 },
  async () => {
    const url = pathToFileURL(resolve('app/fluid-worker.ts')).href;
    const wrapper = `import {parentPort} from 'node:worker_threads';globalThis.self={postMessage:(data,transfer)=>parentPort.postMessage(data,transfer)};await import(${JSON.stringify(url)});parentPort.on('message',data=>self.onmessage({data}));`;
    const worker = new Worker(
      new URL('data:text/javascript,' + encodeURIComponent(wrapper)),
    );
    const base = {
      elapsed: 1 / 60,
      speed: 1,
      paused: false,
      forces: { gravity: 9.8, viscosity: 0.025, agitation: 0 },
      particles: false,
      actions: [],
    };
    const request = (changes = {}, transfers = []) =>
      new Promise((resolve, reject) => {
        const message = (data) => {
          worker.off('error', error);
          if (data.type === 'error') reject(new Error(data.message));
          else resolve(data);
        };
        const error = (e) => {
          worker.off('message', message);
          reject(e);
        };
        worker.once('message', message);
        worker.once('error', error);
        worker.postMessage({ ...base, ...changes }, transfers);
      });
    try {
      const first = await request();
      assert.equal(first.count, 10000);
      assert.ok(first.volume instanceof ArrayBuffer);
      assert.ok(new Float32Array(first.volume).some((v) => v > 1));
      const buffer = first.volume;
      const secondPromise = request(
        { recycleVolume: buffer, actions: [{ type: 'drain', amount: 500 }] },
        [buffer],
      );
      assert.equal(buffer.byteLength, 0);
      const second = await secondPromise;
      assert.equal(second.count, 9500);
      assert.ok(second.volume.byteLength > 0);
      const paused = await request(
        { paused: true, elapsed: 3, recycleVolume: second.volume },
        [second.volume],
      );
      assert.equal(paused.time, second.time);
      assert.equal(paused.volume, null);
      const debug = await request({ paused: true, particles: true });
      assert.equal(debug.positions.byteLength, CAPACITY * 3 * 4);
      assert.equal(debug.time, paused.time);
      const reset = await request(
        {
          paused: true,
          actions: [{ type: 'reset' }],
          recyclePositions: debug.positions,
        },
        [debug.positions],
      );
      assert.equal(reset.count, 10000);
      assert.equal(reset.time, 0);
      assert.ok(reset.volume.byteLength > 0);
      console.log(
        `Worker snapshot: ${first.computeMs.toFixed(1)} ms including physics and volume; ${first.count} particles`,
      );
    } finally {
      await worker.terminate();
    }
  },
);
