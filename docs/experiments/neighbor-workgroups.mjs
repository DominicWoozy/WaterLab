// Benchmark-only scheduling override; shader arithmetic is untouched.
import { fluidShaders } from '../../app/webgpu/fluid-shaders.ts';
import { gridShaders } from '../../app/webgpu/grid-shaders.ts';
import { ComputeKernel } from '../../app/webgpu/compute.ts';

export async function neighborWorkgroups(sim, width = 32) {
  const originalRun = sim.run.bind(sim);
  const original = new Map(sim.kernels);
  const tuned = new Map();
  const shaders = { ...fluidShaders, ...gridShaders };
  for (const name of [
    'lambda',
    'correct',
    'prepareVelocity',
    'viscosity',
    'residual',
    'project',
  ])
    tuned.set(
      name,
      await ComputeKernel.create(
        sim.device,
        `${name} ${width}`,
        shaders[name].replace(
          '@workgroup_size(128)',
          `@workgroup_size(${width})`,
        ),
      ),
    );
  const control = { enabled: true, names: new Set(tuned.keys()) };
  sim.run = (pass, name, p, resources = {}, groups) => {
    const use = control.enabled && control.names.has(name);
    sim.kernels.set(name, use ? tuned.get(name) : original.get(name));
    return originalRun(
      pass,
      name,
      p,
      resources,
      use ? Math.ceil(sim.count / width) : groups,
    );
  };
  return control;
}
