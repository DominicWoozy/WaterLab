// Experimental shader specialization. The application does not import this file.
import { fluidShaders } from '../../app/webgpu/fluid-shaders.ts';
import { gridShaders } from '../../app/webgpu/grid-shaders.ts';
import { PHYSICS_DT } from '../../app/webgpu/simulation.ts';
import { capillaryShaders } from '../../app/webgpu/capillary-shaders.ts';
import { ComputeKernel } from '../../app/webgpu/compute.ts';

export function specialize(code, mode, prune = false, quality) {
  if (mode !== undefined) code = code.replaceAll('P.forces.w', `${mode}.`);
  if (quality !== undefined)
    code = code
      .replaceAll('P.clock.z', `${Math.fround(Math.cbrt(10000 / quality))}f`)
      .replaceAll('P.clock.x', `${Math.fround(PHYSICS_DT)}f`);
  if (prune) {
    // Only remove declarations whose symbol has no references in the module.
    // Preserve all active buffers, binding numbers and access modes.
    const declaration =
      /@group\(0\)\s*@binding\(\d+\)\s*var<(?:uniform|storage)(?:,\s*(?:read|read_write))?>\s+(\w+)\s*:[^;]+;/g;
    for (const match of code.matchAll(declaration)) {
      const rest = code.replace(match[0], '');
      if (!new RegExp(`\\b${match[1]}\\b`).test(rest)) code = rest;
    }
  }
  return code;
}

export async function specializedKernels(sim, options = {}) {
  const { prune = false, constants = true, freeze = false } = options;
  const original = new Map(sim.kernels),
    variants = new Map();
  const started = performance.now();
  const sources = {
    ...fluidShaders,
    ...capillaryShaders,
    ...(freeze ? { prepareVelocity: gridShaders.prepareVelocity } : {}),
  };
  for (const [name, code] of Object.entries(sources)) {
    if (name !== 'viscosity' && !prune && !freeze) continue;
    const modes = name === 'viscosity' && constants ? [0, 1, 2] : [undefined];
    for (const quality of freeze ? [15000, 30000, 50000] : [undefined])
      for (const mode of modes)
        variants.set(
          `${name}/${mode}/${quality}`,
          await ComputeKernel.create(
            sim.device,
            `${name}/${mode}/prune=${prune}`,
            specialize(code, mode, prune, quality),
          ),
        );
  }
  const control = {
    enabled: true,
    variants,
    compileMs: performance.now() - started,
  };
  const modes = new WeakMap(),
    qualities = new WeakMap();
  const originalWrite = sim.writeParameters.bind(sim),
    originalRun = sim.run.bind(sim);
  sim.writeParameters = (slot, input, reset = false) => {
    const p = originalWrite(slot, input, reset);
    modes.set(
      p,
      (input.surfaceTension ?? true)
        ? input.capillaryMode === 'implicit'
          ? 2
          : 1
        : 0,
    );
    qualities.set(p, sim.quality);
    return p;
  };
  sim.run = (pass, name, p, resources = {}, groups) => {
    const mode = name === 'viscosity' && constants ? modes.get(p) : undefined;
    const quality = freeze ? qualities.get(p) : undefined;
    const candidate = variants.get(`${name}/${mode}/${quality}`);
    sim.kernels.set(
      name,
      control.enabled && candidate ? candidate : original.get(name),
    );
    return originalRun(pass, name, p, resources, groups);
  };
  return control;
}
