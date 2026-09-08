# WebGPU migration — 2026-09-08

## Implementation

WebGPU is preferred with 50,000 particles. WebGL2 remains a 15k / 30k fallback;
`?backend=webgl` forces it. This is a full compute-and-render path, not WebGPU
physics copied back to WebGL each frame. JavaScript submits work and handles UI;
production never maps particle or density buffers.

The solver retains three PBF density corrections, two relaxed DFSPH divergence
projections, analytical wall support and two-way duck impulses. It is not full
DFSPH. Count / local scan / totals scan / add / scatter replaces comparison sort
with complete variable-length cell ranges. No fixed bucket capacity truncates
neighbors. A mid-solve rebuild after correction two keeps the maximum outstanding
pressure-plus-contact displacement at a neighbor query below .042×scale, less
than the (.225-.17)×scale cell margin. Impulses follow the same permutation.

Reconstruction rebuilds a grid of final particle positions. The same covariance
regularization, center smoothing, anisotropic kernels, 128×160×96 grid, isovalue
1.15 and three directional filters are retained. Each voxel gathers all eligible
kernels into float32, then the filtered buffer copies to an r32float 3D texture.
The search bound includes kernel stretch and center displacement. Adjacent X
ranges are merged, and 8×4×1 workgroups keep lanes on the same Z slice. Devices
without float32-filterable use equivalent manual eight-tap interpolation.

This does not make reconstruction exactly volume preserving: finite voxel size,
isovalue and smoothing still affect small droplets, thin sheets and surface
position. It does not promise pixel identity with half-float additive splats.
Atomic integer scatter changes summation order and long-running trajectories can
diverge slightly. No integer quantization is used for the density sum. Render
smoothing never writes back into physics positions.

Physics, reconstruction and rendering are separate command buffers submitted in
one ordered queue batch, capped at two frames in flight. No inter-stage CPU waits
are used in the application. Pausing reuses density, particle debug skips density,
and hidden tabs stop submitting. Fixed-step overload limits remain; overloaded
simulation time can slow rather than accumulating an unlimited backlog.

## Validation

`npm run test:webgpu` runs production WGSL on Dawn Metal, retaining the native GPU
instance for the process lifetime. It checks 50k distinct particles, finite and
bounded motion over 240 calm steps, duck quaternion normalization, 120 further
steps with splash/stir/shake, complete sorted cell ranges, 12 density samples
against an independent all-particle CPU sum, empty-field clearing, reinjection,
all three quality levels and two substeps in one command buffer.

The final 50k density sample differences were below 0.000006. Tests permit
max(0.0002, expected×0.0002) to account for floating-point summation. Type checking,
scoped lint and the existing eight WebGL dispatch regressions also pass.

`npm run benchmark:webgpu -- 50000` measures completed whole frames, including
CPU encoding, queue submission and GPU execution; it waits every two frames.
120 physics steps warm the scene, then 80 full frames run, with the first 20
excluded. Output uses 1250×800 RGBA8 plus depth32float, the original 4,212-triangle
duck BVH and a constant 1×1 duck albedo. Reflection and caustics are enabled.
The benchmark reads the final frame and checks water density across front,
middle and back slices; the saved offscreen scene was inspected locally.

Apple M3 Pro / native Metal, final validated run:

| Completed frame time | Milliseconds |
| -------------------- | -----------: |
| Mean                 |        14.54 |
| Median               |        13.07 |
| P95                  |        22.13 |

These are not browser FPS, not a 60 FPS guarantee, and not directly comparable
to the earlier native OpenGL repeated-draw microbenchmarks. Browser command
validation, presentation/compositing and full duck texture sampling add costs;
power state and shader warmup affect results. No browser interaction/visual
automation was performed. Native timestamp experiments were rejected because
queries returned invalid values; reported numbers use completed wall time.

Optional diagnostics: `PROFILE=1` synchronizes after each stage and reports
phase wall times (extra waits change scheduling, do not sum these into browser
FPS); `SPLIT=0` tests a single frame command buffer; `UNFILTERED=1` exercises
manual density interpolation. `webgpu` is a development-only native test
package and is not imported by the deployed application.
