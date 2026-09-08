# WaterLab bottleneck diagnosis — 2026-09-08

Production source: `7fa009615b955c21538c8320e53adb869a953692`.
This investigation adds local diagnostic scripts only; no deployed app behavior
or physics settings changed.

## Method and limits

Apple M3 Pro / native macOS OpenGL, production GLSL adapted only for desktop
version/precision syntax. These are NOT measurements from the user's browser.
The frame probe uses the actual imported BVH/triangles and production camera,
volume, and duck state. Duck albedo uses a constant 1x1 color texture; the probe
renders one fullscreen triangle to RGBA8 without a depth attachment. It verifies
that removing water changes more than 64k pixels in the smaller framebuffer.

`tests/gpu-stage-benchmark.py` replays each configured production draw 16 times
before collecting its query. This reduces Python uniform/texture-binding overhead;
volume replays clear and blend normally. Measurements sum repeated solver passes
per physical step and include duck pressure coupling and reaction reduction.
Frames 0..23 warm up, frames 24..47 are measured.

`tests/gpu-frame-benchmark.py` freezes fluid/duck state at step 240 and alternates
renderer variants, timing 16 repeated draws after parameter setup. 8 warm-up
rounds and 14 measured rounds; reported value is median time per draw.

Repeated draws reuse hot inputs/caches and omit browser command validation,
compositing, synchronization and JS submission. Stage/renderer states also differ.
The numbers identify expensive work; do not sum them into a claimed browser FPS.
They are not directly comparable with the earlier grid benchmark, which includes
uniform and texture setup inside the timed interval. Sub-millisecond differences
can be noisy, particularly at lower resolution.

## Results (milliseconds)

| Stage | 15,000 particles | 30,000 particles |
| --- | ---: | ---: |
| Volume splatting / 3D density generation | 2.685 | 5.849 |
| Physics including duck, excluding grid build | ~0.274 | ~0.813 |
| New grid build | ~0.069 | ~0.175 |
| Anisotropic shape reconstruction | 0.025 | 0.060 |
| Density filtering | 0.004 | 0.027 |
| Complete final composite, 1250x800 | 1.520 | 1.633 |
| Complete final composite, 813x520 | 0.628 | 0.947 |

Renderer ablations at 1250x800:

| Variant | 15,000 | 30,000 |
| --- | ---: | ---: |
| Full production fragment shader | 1.520 | 1.633 |
| Existing reflection toggle off | 1.361 | 1.664 |
| Reflection trace removed from diagnostic shader | 1.173 | 1.428 |
| Duck rendering disabled | 0.512 | 0.908 |
| Thickness density samples removed (ray truncation retained) | 0.948 | 1.197 |
| Entire thickness path replaced by constant | 0.903 | 1.097 |
| Water disabled, scene/duck retained | 0.674 | 0.870 |

Ablations deliberately alter the image to isolate work. Their savings overlap
and must not be added together. Hiding the duck also changes water visibility.
The reflection toggle doesn't consistently save GPU time; source computes the
reflected duck ray before multiplying the resulting color by `reflectionOn`.

## Interpretation

The strongest signal is volume generation. The current 128x160x96 density atlas
contains 1,966,080 voxels. 30k particles submit 600k slice instances / 3.6M vertices.
Each vertex fetches center and three metric textures. Unneeded slices are clipped
in the vertex shader, after these fetches. Surviving fragments overlap and blend
into the density field. Physics and final lighting can be fast while this repeated
volume rasterization stays expensive.

The current reflection is procedural environment shading plus imported duck BVH
ray intersection, not a screen-space reflection march. Removing it is a smaller
win than removing the expensive volume-generation path. Thickness integration
and repeated duck rays are worthwhile secondary improvements.

The application also increases render resolution above 56 FPS. Some saved time
can therefore buy more pixels instead of producing a larger displayed FPS number.

Recommended order:

1. Measure the same stages with asynchronous disjoint timer queries in the actual
   browser before deciding the final optimization budget.
2. Prototype screen-space fluid depth/thickness rendering in WebGL2 to bypass
   3D density splatting, preserving thickness-based absorption and smooth normals.
   Keep the volumetric mode for comparison; view-dependent occlusion needs care.
3. Make reflection-off skip the work and consider reuse of nearby refracted duck
   intersections. These should be separately validated changes.
4. Revisit WebGPU when scaling physics/particle counts further, or when redesigning
   density generation around compute. An API translation alone retains the work.
