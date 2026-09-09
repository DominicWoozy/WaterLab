# Unified primary water

Approved for GitHub push and the existing GitHub Pages deployment on 2026-09-09.
The separate Sites deployment is not part of this release.

## User constraint

Shrink the **rendered water droplets**, not the physical SPH particles. Keep
50,000 main particles, their existing particleMass(), h(), pressure, gravity and
duck coupling. Remove the special thin-film simulation/reconstruction and the
secondary-droplet system rather than merely hiding their UI switches.

## Motion

General cohesion now uses the long-range attractive / short-range repulsive
kernel from [Akinci et al. 2013](https://cg.informatik.uni-freiburg.de/publications/2013_SIGGRAPHASIA_surfaceTensionAdhesion.pdf),
with the authors' [SPlisHSPlasH implementation](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/master/SPlisHSPlasH/SurfaceTension/SurfaceTension_Akinci2013.cpp)
as a reference. This is an adaptation, not a faithful reproduction: normals are
estimated from the existing density gradient, curvature is projected onto the
pair direction, and an empirical scene coefficient / symmetric impulse cap is
used. It is not calibrated against water's SI surface-tension coefficient.

The factor traversal provides normals and density. The existing viscosity
traversal adds cohesion and bounded radial pair damping. Damping is numerical
dissipation to resolve capillary oscillations at the fixed 1/60 s step, rather
than a claim of fully resolved physical viscosity. Pair corrections are central
and symmetric, preserving linear/angular momentum before the pre-existing
clamps, pressure, viscosity and contacts. There is no adhesion force or 2D
thickness / sheet classification. The default adds no new neighbor traversal or
compute pass; the optional fixed-iteration implicit path remains for benchmarks.

Surface tension tends to retract and round a surface. Thin layers must arise
from the imposed stretching / flow and be resolved by the main particles; this
change does not artificially force particles into a sheet.

## One reconstructed surface

Every primary particle contributes to the same anisotropic density field. No
particle is removed from that field because it looks like a drop or sheet.
Overlapping kernels can cross the common isovalue 1.15 and connect naturally.
The existing generic covariance regularization and tangent filtering remain;
there is no flat ellipsoid-patch pass, sheet normal texture or sheet-thickness
accumulation pass.

For isolated primaries the analytic silhouette radius is 0.72 times the original
physical-equivalent sphere radius, a nominal silhouette volume ratio of 0.373248.
It fades out smoothly as neighborhood density increases from 0.18 to 0.70,
letting the continuous field take over. This is one silhouette of an existing
primary, not particle splitting. It prevents a tiny isolated drop disappearing
between coarse voxels. Physical particle volume, total physical water quantity,
neighbor support and time step are unchanged. The rendered isosurface is still
not strictly volume conserving, and the final field can extend beyond a fallback
silhouette; the 37% figure describes the analytic contour, not global water loss.

Limitations: finite voxels/main-particle spacing still limit the thinnest stable
layer and smallest connected neck. Shrinking visual drops alone cannot supply
unresolved physical detail. No claim of micron-scale films or exact coalescence
times is made.

## Verification

- `npm run test:webgpu:surface`: two non-sheet particles attract; close pairs
  repel; bounded damping prevents collapse over 120 steps; disabling tension
  removes the force; lone particles remain inert; pair momentum/torque and
  primary count/IDs are retained; optional implicit path compiles and moves.
- `npm run test:webgpu:details`: actual GPU silhouette radii/coverage shrink;
  main positions/count are unchanged by reconstruction; no sheet/spray resources;
  every primary enters the field; far-pair midpoint density is zero, a close pair
  exceeds the isovalue and is connected; empty state clears both representations.
- `npm run test:webgpu`: full 50k stability, disturbed flow, duck state, density
  CPU reference, all quality levels, injection/drain and two substeps.
- `npm run test:presentation` and `npm run test:gpu`: retain black-frame fix and
  WebGL2 fallback scheduling.

Benchmark: `SCENE=rough PAIRED=1 TENSION=alternate npm run benchmark:webgpu -- 50000`.
This compares the same GPU snapshot with general tension on/off. GPU command
spans and host-completed frame times are reported separately; browser presentation
is excluded. Historical measurements in earlier experiment notes are not claims
about this revision.

Latest local M3 Pro paired run (50,000 particles, 1250 × 800): tension on/off
host-completed mean 21.052 / 20.596 ms, median 20.995 / 21.109 ms, P95
25.287 / 23.249 ms. GPU command-span means 19.315 / 18.872 ms. The mean
increment is about 0.45 ms in this run; the medians overlap, so this is not a
universal overhead guarantee. All GPU validation errors were empty. Do not
compare these directly with the abnormally slow 100+ ms historical session or
claim a browser FPS speedup from that difference. Current local HTTP preview
responded 200; type check, production build and the listed regression suites pass.
