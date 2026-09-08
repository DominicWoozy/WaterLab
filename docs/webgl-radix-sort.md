# WebGL2 stable radix grid build

The particle solver and renderer retain their existing formulas and precision.
The grid builder now uses stable LSD radix-16 instead of fused bitonic sorting.

For each of four 4-bit digits:

1. Count each key's preceding peers with the same digit in its 32-key block.
2. Build 16 bin counts per block, packed into four RGBA32F texels.
3. Run three inclusive prefix-sum passes, with strides 1, 16, and 256 blocks.
4. Scatter one point per key to its unique destination: preceding bins + preceding
   blocks in this bin + its local stable rank.

All keys start in particle-index order. Stable digit passes therefore reproduce
bitonic's `(cell key, particle index)` order exactly, including equal keys.
Live cell keys span 0..30719; padding maps to digit key 65535 but retains its
original 1e9 key value. No fixed per-cell capacity, atomics, floating-point
blending, CPU state readback, or extra WebGL extension is required.
All prefix counts are integers at most 32768, exactly representable in float32.
Three scratch textures use 640 KiB at the current maximum capacity.

Sorting uses 24 draws for either quality, down from 81 (15k) or 94 (30k).
Key generation and cell-range construction add two draws to both methods.
State reordering remains a separate, unchanged MRT pass.
The current grid-key range and 32768 padded capacity are assumptions of this
implementation; expanding either requires updating the digit/scan coverage.

## Measurements, 2026-09-08

Apple M3 Pro, macOS native OpenGL harness compiling the production GLSL with
ES version/precision syntax adaptation. GPU elapsed queries include key creation,
sorting, and cell ranges, but exclude state reordering and subsequent physics.
48 alternating old/new pairs, first 12 discarded; identical initialized state.

| Particles | Previous fused bitonic | Radix | Relative speed |
| --- | --- | --- | --- |
| 15,000 | 3.135 ms | 1.072 ms | 2.93x |
| 30,000 | 3.241 ms | 1.062 ms | 3.05x |

These are GPU grid-build timings, not browser FPS or end-to-end speedups.
No browser visual/compatibility test was performed. Results can differ by driver,
browser and GPU; short timings also include substantial fixed overhead.

## Validation

- `python3 tests/gpu-radix.test.py 30000`: production GPU output against CPU stable
  sorting and previous GPU bitonic, every cell range, no missing/duplicate keys.
  Fixtures cover 0/1/31/32/33/15000/16384/16385/30000 particles, random positions,
  all particles in one cell, wall corners, and shrinking/growing capacity.
- `python3 tests/gpu-fluid-native.py 15000` and `30000`: fluid stability, injection,
  volume reconstruction, coupled state permutation, corrected-neighbor search
  against brute force, divergence projection and rigid-flow preservation.
- `python3 tests/gpu-duck.test.py 15000` and `30000`: settling, wave response,
  draining, and linear/angular impulse conservation.
- `npm run test:gpu`: WebGL command dispatch, point count, digit coverage,
  viewport selection, pause/reset/drain/injection and texture aliasing contracts.
- `npx tsc --noEmit`, lint of changed TypeScript/JS files, and GitHub Pages build.

The existing neighbor probe had stopped removing wall density when duck support
was added to the production expression. It now removes both boundary terms and
asserts every probe substitution matches, so future shader changes fail clearly.
Repository-wide lint still reports existing issues in UI components, hooks and
`app/page.tsx`; none of those files were changed in this optimization.
