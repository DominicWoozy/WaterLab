// Experimental cell-cooperative gather. Not imported by the application.
import {
  common,
  NEIGHBOR_CACHE_BASE,
  recordNeighbor,
} from '../../app/webgpu/common.ts';
import { tensionCommon } from '../../app/webgpu/surface-tension-shaders.ts';
import { ComputeKernel, buffer } from '../../app/webgpu/compute.ts';

export async function sharedNeighbors(sim, width = 32, shared = true) {
  const device = sim.device;
  const tasks = buffer(
    device,
    'shared neighbor tasks',
    (sim.state.size / 48) * 16,
  );
  const indirect = buffer(
    device,
    'shared neighbor dispatch',
    16,
    GPUBufferUsage.STORAGE |
      GPUBufferUsage.INDIRECT |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  );
  device.queue.writeBuffer(indirect, 0, new Uint32Array([0, 1, 1, 0]));
  const taskKernel = await ComputeKernel.create(
    device,
    'neighbor tasks',
    common +
      `
@group(0) @binding(3) var<storage,read> starts:array<u32>;
@group(0) @binding(9) var<storage,read_write> tasks:array<vec4u>;
@group(0) @binding(10) var<storage,read_write> dispatch:array<atomic<u32>>;
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u){
 let c=gid.x;if(c>=30720u){return;}
 let begin=starts[c];let end=starts[c+1u];let count=(end-begin+${width - 1}u)/${width}u;
 if(count==0u){return;}let slot=atomicAdd(&dispatch[0],count);
 for(var k=0u;k<count;k++){tasks[slot+k]=vec4u(c,begin+k*${width}u,min(end,begin+(k+1u)*${width}u),0u);}
}`,
  );
  const declaration =
    common +
    tensionCommon +
    `
@group(0) @binding(1) var<storage,read> input:array<Particle>;
@group(0) @binding(2) var<storage,read_write> output:array<Particle>;
@group(0) @binding(3) var<storage,read_write> starts:array<u32>;
@group(0) @binding(5) var<storage,read_write> factors:array<vec4f>;
@group(0) @binding(6) var<storage,read> duck:Duck;
@group(0) @binding(8) var<storage,read_write> surface:array<vec4f>;
@group(0) @binding(9) var<storage,read> tasks:array<vec4u>;
var<workgroup> task:vec4u;
var<workgroup> interval:vec2u;
var<workgroup> positions:array<vec4f,${width}>;
`;
  const shader = (velocity) =>
    declaration +
    `
@compute @workgroup_size(${width}) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 if(lane==0u){task=tasks[group.x];}
 let t=workgroupUniformLoad(&task);
 let i=t.y+lane;let validLane=i<t.z;
 var p=vec3f(0.);if(validLane){p=input[i].pos.xyz;}
 let base=vec3i(i32(t.x%32u),i32((t.x/32u)%40u),i32(t.x/1280u));
 let wall=wallSupport(p)+duckSupport(p,duck);
 var rho=wall.w;var sum=0.;var grad=wall.xyz;var count=0u;
 for(var dz=-1;dz<=1;dz++){let z=base.z+dz;if(z<0||z>=24){continue;}
  for(var dy=-1;dy<=1;dy++){let y=base.y+dy;if(y<0||y>=40){continue;}
   if(lane==0u){let row=32*(y+40*z);interval=vec2u(starts[u32(row+max(0,base.x-1))],starts[u32(row+min(31,base.x+1)+1)]);}
   let bounds=workgroupUniformLoad(&interval);
   for(var begin=bounds.x;begin<bounds.y;begin+=${width}u){
    ${
      shared
        ? `if(begin+lane<bounds.y){positions[lane]=input[begin+lane].pos;}
    workgroupBarrier();`
        : ''
    }
    if(validLane){
     for(var k=0u;k<min(${width}u,bounds.y-begin);k++){
      let j=begin+k;if(j==i){continue;}
      let diff=p-${shared ? 'positions[k]' : 'input[j].pos'}.xyz;let r2=dot(diff,diff);
      if(r2>=h()*h()||r2<1e-12){continue;}
      let r=sqrt(r2);let q=1.-r/h();let gradient=(2.*q/(h()*REST*r))*diff;
      rho+=q*q;grad+=gradient;sum+=dot(gradient,gradient);
      ${recordNeighbor()}
     }
    }
    ${shared ? 'workgroupBarrier();' : ''}
   }
  }
 }
 if(!validLane){return;}
 starts[${NEIGHBOR_CACHE_BASE}u+i]=count;
 ${
   velocity
     ? `
 surface[i]=vec4f(limited(-h()*grad,2.),(rho+1.)/REST);
 var f=0.;if(count>=12u&&rho>REST*.4){f=1./max(sum+dot(grad,grad),1e-6);}
 factors[i]=vec4f(f,rho/REST,f32(count),surfaceWeight(surface[i]));
 var a=input[i];var v=(a.pos.xyz-a.old.xyz)/P.clock.x;
 if(a.old.w<.5){v=vec3f(0.,-1.4,0.);}
 a.vel=vec4f(limited(v,12.)*pow(.998,P.clock.x*60.),a.vel.w);output[i]=a;
 `
     : `factors[i]=vec4f(-.25*max(rho/REST-1.,0.)/(sum+dot(grad,grad)+2.),rho,0.,0.);`
 }
}`;
  const kernels = new Map();
  for (const [name, velocity] of [
    ['lambda', false],
    ['prepareVelocity', true],
  ])
    kernels.set(
      name,
      await ComputeKernel.create(device, 'shared ' + name, shader(velocity)),
    );
  const bindings = new Map();
  const originalRun = sim.run.bind(sim),
    originalGrid = sim.buildGrid.bind(sim);
  const control = {
    enabled: true,
    tasks,
    indirect,
    kernels,
    taskKernel,
    velocity: true,
    pressure: true,
  };
  let fresh = false;
  control.resetFresh = () => {
    fresh = true;
  };
  sim.buildGrid = (encoder, p = sim.parameters[12], reorder = false) => {
    originalGrid(encoder, p, reorder);
    fresh = true;
    if (!control.enabled) return;
    encoder.clearBuffer(indirect, 0, 4);
    const pass = encoder.beginComputePass({ label: 'neighbor task list' });
    taskKernel.dispatch(
      pass,
      { 0: p, 3: sim.starts, 9: tasks, 10: indirect },
      240,
    );
    pass.end();
  };
  sim.run = (pass, name, p, resources = {}, groups) => {
    const use =
      control.enabled &&
      ((name === 'lambda' && fresh && control.pressure) ||
        (name === 'prepareVelocity' && control.velocity));
    if (name === 'lambda') fresh = false;
    if (!use) return originalRun(pass, name, p, resources, groups);
    const kernel = kernels.get(name);
    const r = {
      0: p,
      1: sim.state,
      2: sim.spare,
      3: sim.starts,
      5: sim.lambda,
      6: sim.duck,
      8: sim.surface,
      9: tasks,
      ...resources,
    };
    const key = [name, ...kernel.bindings.map((b) => r[b])];
    let group;
    // Benchmark only: bind-group cache keyed by actual buffer identity.
    for (const [k, v] of bindings)
      if (k.every((x, i) => x === key[i])) {
        group = v;
        break;
      }
    if (!group) {
      group = device.createBindGroup({
        layout: kernel.layout,
        entries: kernel.bindings.map((binding) => ({
          binding,
          resource: { buffer: r[binding] },
        })),
      });
      bindings.set(key, group);
    }
    pass.setPipeline(kernel.pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(indirect, 0);
  };
  return control;
}
