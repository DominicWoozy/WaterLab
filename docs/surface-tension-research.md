# Surface-tension performance assessment

> Historical experiment: superseded locally by [unified primary water](unified-primary-water.md).
> The current app has no sheet-specific physics/render pass and no secondary droplets.


Local implementation and research; do not push or deploy without a new user request.
The current bounded implicit experiment is documented in [thin-film physics](thin-film-physics.md).

## Preferred next candidate: Jeske et al.

[Implicit Surface Tension for SPH Fluid Simulation (2023)](https://srjeske.de/publications/2023-tog-sph-surface-tension/)
([author PDF](https://srjeske.de/publications/2023-tog-sph-surface-tension/JWL%2B23.pdf))
uses a linearized implicit cohesion model and can combine it with implicit
viscosity. Its useful optimizations are:

- Precompute a per-particle scalar to avoid neighbor-of-neighbor work.
- Evaluate the linear operator without assembling a global matrix.
- Warm-start conjugate gradient from the previous velocity increment.
- Reuse the matrix structure when coupling viscosity and tension.

The authors report neighborhoods around 30–40 particles and larger stable time
steps in demanding tension cases. Their CPU comparisons include SIMD
optimizations absent from some baselines; these are not direct WebGPU speedup
predictions. At low tension a simple explicit method may remain cheaper. The
paper's diagonal preconditioner did not produce a tangible improvement in their
tests; adding one is not an automatic optimization.

For WaterLab we now use a frozen-coefficient backward-Euler approximation of our
existing capillary kernel. It precomputes the force/right-hand side and inverse
diagonal, then performs four matrix-free Jacobi sweeps. Only particles classified
as sheets traverse neighbors in these sweeps. The positive cohesion graph is
diagonally dominant; the repulsive core remains explicit. No global reduction or
CPU readback is needed, and iteration cost has a fixed upper bound.

This borrows the local-coefficient / implicit-operator structure, **not** the
paper's full density derivative, implicit viscosity or warm-started CG method.
It does not increase the fixed simulation time step or establish a speedup over
our cheaper explicit baseline. The latter remains available to native benchmarks.
GPU tests evaluate the actual linear residual against a CPU operator reference:
on a moving stretched-sheet fixture the relative max residual was 1.45e-7 and
maximum Jacobi row contraction 0.000618. Those are fixture results, not a proof
of that tolerance for every possible injected state.

Coefficients are regenerated after particle reorder. We intentionally do not
warm-start from a buffer indexed by stale sorted slots.

## More invasive candidate: Probst and Teschner

[Unified Pressure, Surface Tension and Friction for SPH Fluids](https://doi.org/10.1145/3708034)
([author PDF](https://cg.informatik.uni-freiburg.de/publications/2024_TOG_unifiedPressureSurfaceTensionFriction_v3.pdf))
solves the coupled forces together and uses NNCG acceleration to reduce iterative
work. The authors explicitly note that larger time steps can require enough
extra iterations to erase the benefit. They recommend a wider support for
tension than pressure, with roughly 100 versus 30 neighbors in their 3D setup.
Their showcased implementation uses CPU neighbor search and OpenCL on RTX 4090;
its timings cannot be transferred to a browser on Apple Silicon.

Engineering assessment: useful for interface-force consistency and solver
acceleration research, but a full replacement is unlikely to be the lowest-cost
next change to this 50k-particle web demo. NNCG is not a plug-in change to the
current PBF/DFSPH split without deriving the appropriate coupled system.

## Lower-risk baseline and later scheduling work

[Akinci et al., Versatile Surface Tension and Adhesion for SPH Fluids](https://cg.informatik.uni-freiburg.de/publications/2013_SIGGRAPHASIA_surfaceTensionAdhesion.pdf)
combines cohesion and curvature-related forces. Its explicit form is a useful
baseline, but stronger tension/smaller particles can demand shorter steps. Our
current regularized pair kernel is an approximation, not a reproduction.

[Asynchronous Liquids: Regional Time Stepping for Faster SPH and PCISPH](https://arxiv.org/abs/2009.14514)
adapts effort spatially to local dynamics. It motivates keeping fast spray from
forcing small global steps. It does not justify merely skipping every other
tension update: interface exchanges and neighboring time levels must stay
consistent. Integrating it with our split solver is a separate substantial task.

## Evaluation decision

Keep the existing fused neighbor traversals as the performance baseline. Before
replacing them, measure GPU timestamps for classification, pressure, tension,
density and final rendering separately; queue completion measures and browser
FPS conflate these costs. Use droplet pressure/oscillation, a retracting sheet and
a disturbed 50k tank to evaluate accuracy, not just whether the result looks wet.

A reasonable product target is less than about 1 ms additional completed-frame
cost on the current reference machine, treated as a budget rather than a
promise. Compare median/P95 and solve residual at equal simulated time. Simply
weakening tension, clamping forces harder or limiting iterations without checking
error can appear faster while changing the physics. No paper establishes that
implicit tension is universally faster than the current explicit approximation.

## Periodic black-frame fix

The reported whole-screen periodic flash matches both engines changing
canvas.width/height in their FPS-feedback block after rendering. Resizing clears
the displayed drawing buffer. Both engines now queue resize requests and apply
them before drawing the next frame; WebGPU also defers them when hidden or when
GPU backpressure skips a frame. WebGL avoids same-size assignments as well.
`npm run test:presentation` exercises the real animation loops with a canvas that
loses its picture when resized, covering slow/fast cadence, observer events,
pause, hidden frames and GPU backpressure. This is a scheduling regression test,
not a browser visual confirmation or a change to the physical time step.
