# Local thin-film capillary experiment

> Historical experiment: superseded locally by [unified primary water](unified-primary-water.md).
> The current app has no sheet-specific physics/render pass and no secondary droplets.


This change is local only. Do not push or publish it without a new user request.

The WebGPU physics panel has “薄膜张力与破裂”, enabled by default. It is
independent of the lighting panel's detail reconstruction switch. Try the rough
preset or stirring, then reset before comparing trajectories with the switch off.
WebGL2 retains the previous solver and disables the new switch.

## Model and scheduling

The existing divergence-factor neighbor traversal estimates covariance, planar
coverage and a 2D kernel number density. With W(r) = 10/(pi h²) (1-r/h)³,
area A = 1/sum W and thickness t = particleVolume/A. Volume is fixed, so
stretching reduces estimated thickness; compression increases it. Thickness is
estimated from current positions rather than a history field or an independent
lubrication-equation solve.

The explicit baseline adds central capillary pair accelerations in the existing
viscosity traversal. The default now applies a bounded implicit approximation.
The long-range coefficient is derived from the two-sided sheet energy
E = 2 sigma sum A. Symmetric sheet coverage, a short-range repulsive core and
an equal/opposite pair limit regularize unresolved collapse. This is consequently
not an exact global energy minimization. sigma/rho = 0.000072 is the water-like
coefficient with metre-scale scene coordinates; the complete application has
not been calibrated against a physical experiment.

The explicit central pair forces conserve linear and angular momentum before the existing velocity
clamps, viscosity, pressure and contact stages. A symmetric neighbor-count limit
bounds the added speed to .12 per substep. The wall/duck support layer is excluded
to avoid introducing adhesion. No extra neighbor-grid build or CPU particle readback is needed. An 800 kB GPU
buffer holds area, coverage, count and thickness, recomputed after reordering.
The implicit path adds prepare + four Jacobi + apply dispatches and 2.4 MB for
RHS/velocity buffers. It solves (I + dt² L) v = v_base + dt f with a frozen positive
cohesion Laplacian L; the repulsive core stays explicit. Only active sheet
particles traverse neighbors. Four sweeps avoid unbounded solver work and GPU
reductions, but this path is not inherently faster than the explicit baseline.
See [surface-tension research](surface-tension-research.md) for the paper and
implementation differences.

As extensional flow separates a thin neck beyond support radius h, its physical
connections disappear. Existing holes can retract into their surrounding rim.
No particles are randomly deleted or hidden. Reconstruction patches are limited
to .49h in this mode, preserving nominal ellipsoid volume while preventing
sheet patches from bridging a gap larger than the interaction support.

This remains a coarse capillary breakup experiment. It does not implement
adaptive splitting/merging, van der Waals nucleation, surfactants, air coupling,
full thin-film Navier–Stokes, or resolution-independent breakup/drop sizes.
A perfectly uniform sheet is not perforated by an artificial random timer.
Finite sheet boundaries and the repulsive core affect long-time rim motion;
Taylor–Culick retraction speed has not been quantitatively validated.

## Oversized droplets

The volume gather previously used a fixed world-space minimum radius .095.
This stopped sparse kernels shrinking at high particle counts. All kernel radii
now scale with particle resolution. Sparse spray with a few distant neighbors
also takes the analytic drop path (rho < .65, at most 10 neighbors), while
planar neighborhoods can still become sheets. This avoids a second coarse
voxel surface inflating the same droplets. Each analytic drop still represents
its nominal particle volume. The separate secondary-droplet option below
provides smaller visual droplets without refining the main SPH discretization.

## Secondary droplets and continuous sheet shading

The lighting panel adds “细小飞溅”. Fast, sparse airborne particles can each own
8 secondary droplets; the shared pool holds at most 256 packs / 2,048 children.
After a 0.06 s fade-in, 90% of the parent's nominal displayed volume is shared
among the eight children. Their radius is approximately 0.483 of the original;
the retained parent's radius is approximately 0.464. They use gravity, linear
air drag and damped wall / duck-proxy contacts, and retire after 0.6 s (0.15 s blend-out) or
when the parent rejoins a sheet / body. Still water emits nothing.

The parent remains a full-mass coarse SPH coupling proxy. Only displayed volume
is partitioned: this is one-way secondary motion, not strict two-way mass,
momentum or energy exchange with the fluid or duck. The initial symmetric
separation velocity approximates unresolved breakup energy. Nominal analytic
parent + child volumes sum to one parent volume; the voxel isosurface is not
strictly volume conserving. Reset, draining, pause and the UI switch reclaim
packs. Ownership tags travel in pos.w through sorting and prediction/correction;
generation tags prevent reuse collisions. No persistent state is keyed by vel.w,
which can contain duplicate nominal IDs after injection.

Thin sheets now accumulate compact weighted normals from overlapping analytic
patches in the existing additive-thickness pass (one additional RGBA16F target).
Weights fade at each footprint edge and reject incompatible depth/normal layers.
The final surface uses this shared normal instead of five nearest-pixel taps.
It leaves coverage/hole geometry and additive optical thickness unchanged. On
the curved-sheet fixture, summed normal-direction error fell to 0.229 of the
nearest-patch baseline over 1,637 interior samples. This removes normal seams;
it is not a continuous mesh or sub-particle thin-film solver. Resolution still
limits sheet rim scalloping and the smallest physically resolved thickness.

## Verification

- `npm run test:webgpu:spray`: pool capacity, half-radius droplets, nominal volume
  partition, duplicate IDs/reorder, lifetime, pause, switch, reset and drain.
- `npm run test:webgpu:film`: implicit operator/RHS/residual; planar interior balance; equal/opposite impulses
  and torque; volume = area × thickness; thinning under 20% planar stretch;
  zero added motion when disabled; wall exclusion; isolated drops; physical
  hole expansion over three seconds; a stretched 513-particle neck becoming
  disconnected while retaining every particle ID and volume.
- `npm run test:webgpu`: 50k stable and disturbed motion, duck coupling, exact
  grid/density reference, all supported counts, empty/reinjection, two substeps.
- `npm run test:webgpu:details`: actual offscreen depth/coverage, optical
  thickness reference, holes, drop volume and a six-particle sparse spray
  regression that previously entered the inflated coarse volume.
- `npm run test:gpu`: existing WebGL2 scheduling regressions.

Performance comparison: `SPRAY=0 PAIRED=1 SCENE=rough CAPILLARY=alternate npm run benchmark:webgpu -- 50000`.
Implicit/on and explicit/off modes restore the same GPU-resident particle and duck snapshot and include
copy, physics, density and rendering. Native Metal measurements exclude browser
presentation/compositing and are not a browser FPS guarantee. Use
`SCENE=rough npm run benchmark:webgpu -- 50000` for evolving secondary-droplet
lifecycles; paired snapshot restoration deliberately disables spray.

### Measurement limits on this run

At 50,000 particles / 1250 × 800 on the current M3 Pro session, the paired
implicit/explicit run measured median 91.46 / 91.53 ms (mean 102.34 / 104.49 ms).
A separate clean published-HEAD control also ran slowly: median 129.35 ms,
mean 115.16 ms. The final evolving scene with all effects measured mean 116.81 ms
and GPU timestamp span mean 115.20 ms; all GPU validation error lists were empty.
These runs show large variability and **do not validate smooth 50k browser FPS**.
The published-HEAD control has different surface classification, so it is a
runtime diagnostic rather than an exact-state visual-quality comparison. No
speedup or reliable incremental spray-cost claim is made from these numbers.
The benchmark now reports GPU command spans as well as host-completed frame time.

Type checking, native physics/detail/spray tests and presentation/WebGL2 regressions
pass. Scoped engine lint passes; page lint still reports the three pre-existing
React ref/effect and internal-anchor findings, outside this change.

## References

[Wang et al., Thin-Film Smoothed Particle Hydrodynamics Fluid](https://arxiv.org/abs/2105.07656)
motivates evolving thickness with an in-plane particle distribution. This
implementation is a limited capillary approximation, not a reproduction.

[Akinci et al., Versatile Surface Tension and Adhesion for SPH Fluids](https://cg.informatik.uni-freiburg.de/publications/2013_SIGGRAPHASIA_surfaceTensionAdhesion.pdf)
motivates regularized pair cohesion; the kernel here is different.

[Taylor–Culick retractions and the influence of the surroundings](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/taylorculick-retractions-and-the-influence-of-the-surroundings/8F245EC9BB4BC42D887271686136D59A)
is a future quantitative validation target, not an accuracy claim for this version.
