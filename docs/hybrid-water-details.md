# Hybrid droplets and thin sheets — 2026-09-08

> Historical experiment: superseded locally by [unified primary water](unified-primary-water.md).
> The current app has no sheet-specific physics/render pass and no secondary droplets.


The original rendering experiment below has a subsequent local-only
[capillary physics and droplet-size update](thin-film-physics.md). Its optional
physics mode limits sheet support to .49h, and the current sparse-drop classifier
and volume-kernel scaling are documented there. The original measurements below
refer to the first published rendering implementation.

The WebGPU lighting panel now has “精细水滴与薄片”, enabled by default.
Turning it off rebuilds the original volume even when paused; particle debug
still skips reconstruction, and WebGL2 keeps its original renderer.

## Reconstruction

The geometry compute pass reuses its existing neighborhood covariance, density
and neighbor count. Sparse particles become analytic droplets. Nearly planar
neighborhoods with sufficient tangential spread become sheet patches; a
cofactor-based normal and variance test avoids a second neighbor search or a
full eigensolver. Other particles keep the existing volumetric kernel.

Classified particles append one render proxy each into a GPU buffer with room
for every physical particle. GPU-generated indirect arguments draw only these
proxies. No CPU readback, additional physics particles or truncated neighbor
buckets are introduced. Their contributions are excluded from the coarse
volume and its height bound, so a detached drop no longer requires a minimum
voxel kernel radius or makes the bulk ray marcher traverse its empty altitude.

Droplets use radius (3 × particleVolume / 4π)^(1/3). Sheet proxies use tangential
radius .78h and a thin radius that gives each ellipsoid the same nominal volume
as its source particle. This is a local reconstruction, not a globally
volume-preserving union of ellipsoids. The physical solver is unchanged.

A first raster pass solves ray/ellipsoid intersections and stores nearest depth,
normal, ID and analytic edge coverage. It evaluates the discriminant from the
closest point rather than subtracting large quadratic terms. This matters for
thin sheets. The quad is padded by a pixel, but the world-space radius is not
enlarged for visibility. Sheet normals get four additional screen-space taps,
rejecting empty pixels, depth jumps and incompatible normals. Holes are not
filled by the filter.

A second, additive R16F pass accumulates overlapping sheet chords within .08
world units of the nearest sheet and with compatible normals. Using just the
frontmost chord would discard most of the represented optical thickness.
View thickness is corrected for the entry/refraction angle before Beer–Lambert
absorption. Droplets use their own analytic refracted chord and exit refraction.
The main composite selects detail versus bulk surface by depth and preserves
primary duck occlusion, Fresnel, absorption and the original lighting toggles.

## Scope and limits

This adds render detail, not adaptive physical resampling, surface-tension
breakup or secondary mist. Very thin physical structures absent from the
particle distribution cannot be recovered reliably. Classification can switch
as neighborhoods change; no persistent hysteresis is claimed.

The detail pass stores the nearest visible layer. Secondary reflection rays
still intersect the environment/duck, not every detail proxy; arbitrary layers
of overlapping transparent sheets are not fully traced. Sheet geometry and
additive thickness are separate approximations, and blending accumulates in
half precision. These are deliberate limits of this experimental, bounded-cost
path. The UI switch makes comparison and rollback immediate.

## Verification

`npm run test:webgpu:details` renders a curved 916-particle sheet with a hole and
five isolated drops. It checks the full indirect count, finite proxy data,
per-proxy nominal volume, unchanged physical positions through grid reorder,
hole coverage, visible smooth drop footprints, disabling the effect, and empty
reset. The sheet thickness sample is compared with an independent CPU sum of
ray/ellipsoid intersections, including the layer/normal rejection criteria.
In the validated run, summed thickness was 0.119873 versus 0.120087 from the
CPU reference; the nearest chord alone was only 0.013138. The before/after
native offscreen images were inspected. Existing WebGPU numeric and eight
WebGL scheduling regressions also pass; browser UI automation was not run.

For a paired performance check:

```sh
PAIRED=1 SCENE=rough DETAILS=alternate npm run benchmark:webgpu -- 50000
```

A turbulent seed is generated first. Each measured frame restores the same
GPU-resident particle/duck snapshot, runs one physical step and the full render,
and alternates the feature every two frames. Restoring the snapshot adds the
same GPU copy cost to both modes; it avoids comparing different water states.
The normal benchmark still advances time continuously when PAIRED is absent.

M3 Pro native Metal, 50k particles, 1250×800, 1,489 detail proxies:

| Mode | Mean frame ms | Median |   P95 |
| ---- | ------------: | -----: | ----: |
| Off  |         18.58 |  18.45 | 19.83 |
| On   |         19.24 |  19.24 | 20.44 |

The measured difference is about 0.67 ms (3.6%), not zero. These are one-machine
offscreen measurements with the original duck mesh and a 1×1 test albedo,
excluding browser presentation/compositing. They are not a browser FPS promise.
Manual trilinear sampling without float32-filterable is also exercised.

Design references: [Yu and Turk, anisotropic kernels](https://faculty.cc.gatech.edu/~turk/my_papers/sph_surfaces.pdf),
[van der Laan et al., screen-space fluid rendering](https://wstahw.win.tue.nl/edu/2IV06/andrei/particle_rendering/provided/p91-van_der_laan.pdf).
This is a focused hybrid implementation, not a reproduction of either paper.
