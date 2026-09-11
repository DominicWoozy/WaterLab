// Test the production transport loop against analytic Snell-law fixtures.
// Density is analytic here to isolate interface transport from reconstruction.
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { transmissionTrace } from '../app/webgpu/refraction-shaders.ts';
import { detailCommon } from '../app/webgpu/detail-shaders.ts';
import { ComputeKernel } from '../app/webgpu/compute.ts';
Object.assign(globalThis, globals);
globalThis.nativeGPU = create(['backend=metal']);
const adapter = await nativeGPU.requestAdapter();
assert.ok(adapter, 'Metal adapter required');
const device = await adapter.requestDevice();
const errors = [];
device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
const code =
  detailCommon +
  `
struct Scene {config:vec4f}
var<private> S:Scene;
var<private> mode:u32;
@group(0) @binding(0) var<storage,read> details:array<Detail>;
@group(0) @binding(1) var<storage,read> queries:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> results:array<vec4f>;
fn density(p:vec3f)->f32 {
 if(mode>=7u){let gap=select(0.,.02,mode==8u);return select(0.,2.,abs(p.x)<1.86-gap && abs(p.z)<1.36-gap && p.y>-.96+gap && p.y<2.);}
 if(mode==6u){return select(0.,2.,dot(p,p)<.04);}
 if(mode>=5u){return 0.;}
 var bottom=-.2;
 if(mode==2u){bottom=-.25+.25*p.x;}
 let second=(mode==1u && p.y<-.3 && p.y>-.5)||(mode==4u && p.y<-.3 && p.y>-.32);
 return select(0.,2.,(p.y<0. && p.y>bottom)||second);
}
fn normalAt(p:vec3f)->vec3f {
 if(mode>=7u){let gap=select(0.,.02,mode==8u);if(abs(p.x)>1.85-gap){return vec3f(sign(p.x),0.,0.);}if(abs(p.z)>1.35-gap){return vec3f(0.,0.,sign(p.z));}return vec3f(0.,-1.,0.);}
 if(mode==6u){return normalize(p);}
 if(mode==2u && p.y<-.05){return normalize(vec3f(.25,-1.,0.));}
 if(abs(p.y)<.01 || abs(p.y+.3)<.01){return vec3f(0.,1.,0.);}
 return vec3f(0.,-1.,0.);
}
fn boxHit(ro:vec3f,rd:vec3f)->vec2f {
 let safe=select(vec3f(.00001),rd,abs(rd)>vec3f(.00001));
 let a=(vec3f(-2.,-1.,-2.)-ro)/safe;let b=(vec3f(2.,1.,2.)-ro)/safe;
 return vec2f(max(max(min(a,b).x,min(a,b).y),min(a,b).z),min(min(max(a,b).x,max(a,b).y),max(a,b).z));
}
fn lightReceiver(ro:vec3f,rd:vec3f)->vec2f {if(mode>=7u && rd.y<0.){return vec2f((-.97-ro.y)/rd.y,1.);}return vec2f(1e5,0.);}
fn duckTrace(ro:vec3f,rd:vec3f,end:f32)->vec4f {
 if(mode==3u && rd.y<0.){return vec4f(min(end,(-.1-ro.y)/rd.y),0.,0.,0.);}
 return vec4f(end,-1.,0.,0.);
}
` +
  transmissionTrace +
  `
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;let q=queries[2u*i];let d=queries[2u*i+1u];mode=u32(q.w);S.config=vec4f(1.);
 let p=traceTransmission(q.xyz,d.xyz,true,u32(d.w));
 results[3u*i]=vec4f(p.direction,p.distance);
 results[3u*i+1u]=vec4f(p.origin,p.weight);
 results[3u*i+2u]=vec4f(f32(p.events),f32(p.complete),0.,0.);
}`;
const kernel = await ComputeKernel.create(device, 'refraction fixtures', code);
const norm = (v) => {
  const l = Math.hypot(...v);
  return v.map((x) => x / l);
};
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const refract = (d, n, eta) => {
  const c = dot(d, n),
    k = 1 - eta * eta * (1 - c * c);
  assert.ok(k >= 0);
  return d.map((x, i) => eta * x - (eta * c + Math.sqrt(k)) * n[i]);
};
const incident = [
  Math.sin((50 * Math.PI) / 180),
  -Math.cos((50 * Math.PI) / 180),
  0,
];
const water = refract(incident, [0, 1, 0], 1 / 1.333);
const entry = [-0.1, Math.sqrt(0.03), 0],
  outward = norm(entry);
const dropDirection = refract([0, -1, 0], outward, 1 / 1.333);
const dropStart = entry.map((x, i) => x - outward[i] * 0.00002);
const fixtures = [
  { name: 'parallel slab', mode: 0, p: [0, -0.0005, 0], d: water },
  { name: 'two detached slabs', mode: 1, p: [0, -0.0005, 0], d: water },
  { name: 'wedge', mode: 2, p: [0, -0.0005, 0], d: water },
  { name: 'opaque inside water', mode: 3, p: [0, -0.0005, 0], d: water },
  { name: 'thin second layer', mode: 4, p: [0, -0.0005, 0], d: [0, -1, 0] },
  {
    name: 'total internal reflection',
    mode: 0,
    p: [0, -0.1, 0],
    d: [Math.sin(Math.PI / 3), -0.5, 0],
  },
  { name: 'analytic drop', mode: 5, p: dropStart, d: dropDirection, detail: 1 },
  {
    name: 'analytic drop TIR',
    mode: 5,
    p: [0, 0.18, 0],
    d: [1, 0, 0],
    detail: 1,
  },
  {
    name: 'density-reconstructed drop',
    mode: 6,
    p: entry.map((x, i) => x - outward[i] * 0.0005),
    d: dropDirection,
  },
];
const contactStart = fixtures.length;
fixtures.push(
  {
    name: 'tank right contact at TIR angle',
    mode: 7,
    p: [1.8, -0.3, 0],
    d: [0.5, -0.5, Math.sqrt(0.5)],
  },
  {
    name: 'tank left contact at TIR angle',
    mode: 7,
    p: [-1.8, -0.3, 0],
    d: [-0.5, -0.5, Math.sqrt(0.5)],
  },
  {
    name: 'tank front contact at TIR angle',
    mode: 7,
    p: [0, -0.3, 1.3],
    d: [Math.sqrt(0.5), -0.5, 0.5],
  },
  {
    name: 'tank back contact at TIR angle',
    mode: 7,
    p: [0, -0.3, -1.3],
    d: [Math.sqrt(0.5), -0.5, -0.5],
  },
  {
    name: 'tank floor contact at TIR angle',
    mode: 7,
    p: [0, -0.9, 0],
    d: [Math.sqrt(0.75), -0.5, 0],
  },
  {
    name: 'detached surface close to wall',
    mode: 8,
    p: [1.78, -0.3, 0],
    d: [0.8, -0.6, 0],
  },
  {
    name: 'detached surface above floor',
    mode: 8,
    p: [0, -0.9, 0],
    d: [0.6, -0.8, 0],
  },
  {
    name: 'airborne surface above wall top',
    mode: 7,
    p: [1.8, 0.9, 0],
    d: [0.8, 0, 0.6],
  },
);
const make = (size, usage) => device.createBuffer({ size, usage });
const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const query = make(fixtures.length * 32, storage);
device.queue.writeBuffer(
  query,
  0,
  new Float32Array(
    fixtures.flatMap((f) => [...f.p, f.mode, ...f.d, f.detail || 0]),
  ),
);
const details = make(64, storage);
device.queue.writeBuffer(
  details,
  0,
  new Float32Array([0, 0, 0, 1, 25, 0, 0, 0.2, 0, 25, 0, 0, 0, 0, 25, 1]),
);
const output = make(fixtures.length * 48, storage | GPUBufferUsage.COPY_SRC);
const read = make(
  output.size,
  GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
);
const e = device.createCommandEncoder(),
  pass = e.beginComputePass();
kernel.dispatch(pass, { 0: details, 1: query, 2: output }, fixtures.length);
pass.end();
e.copyBufferToBuffer(output, 0, read, 0, output.size);
device.queue.submit([e.finish()]);
await read.mapAsync(GPUMapMode.READ);
const range = read.getMappedRange();
globalThis.retained = [read, range];
const values = new Float32Array(range.slice(0));
read.unmap();
const results = fixtures.map((f, i) => ({
  name: f.name,
  direction: Array.from(values.slice(i * 12, i * 12 + 3)),
  distance: values[i * 12 + 3],
  origin: Array.from(values.slice(i * 12 + 4, i * 12 + 7)),
  weight: values[i * 12 + 7],
  events: values[i * 12 + 8],
  complete: values[i * 12 + 9],
}));
console.log(JSON.stringify(results, null, 2));
const near = (a, b, tol, label) =>
  assert.ok(Math.abs(a - b) < tol, `${label}: ${a} != ${b}`);
const direction = (i, expected) =>
  results[i].direction.forEach((x, k) =>
    near(x, expected[k], 0.002, fixtures[i].name),
  );
for (const r of results) {
  assert.ok(
    [...r.direction, r.distance, r.weight, ...r.origin].every(Number.isFinite),
  );
  near(Math.hypot(...r.direction), 1, 1e-5, r.name);
  assert.ok(r.weight >= 0 && r.weight <= 1);
}
direction(0, incident);
direction(1, incident);
direction(4, [0, -1, 0]);
near(results[0].distance, 0.1995 / -water[1], 0.001, 'slab optical thickness');
near(
  results[1].distance,
  0.399 / -water[1],
  0.002,
  'air gap excluded from absorption',
);
near(results[4].distance, 0.219, 0.002, 'thin-layer absorption');
assert.equal(results[0].events, 1);
assert.equal(results[1].events, 3);
assert.equal(results[4].events, 3);
assert.equal(results[3].events, 0);
assert.equal(results[3].complete, 1);
near(
  results[3].distance,
  0.0995 / -water[1],
  1e-5,
  'opaque receiver truncates water',
);
direction(2, refract(water, norm([-0.25, 1, 0]), 1.333));
for (const i of [5, 7]) {
  assert.equal(results[i].events, 4);
  assert.equal(results[i].complete, 0);
  near(results[i].weight, 1, 1e-6, 'TIR retains energy');
  assert.ok(results[i].distance > 0);
}
const b = dot(dropStart, dropDirection);
const chord = -b + Math.sqrt(b * b - dot(dropStart, dropStart) + 0.04);
const exit = dropStart.map((x, i) => x + chord * dropDirection[i]);
direction(
  6,
  refract(
    dropDirection,
    norm(exit).map((x) => -x),
    1.333,
  ),
);
near(results[6].distance, chord, 1e-5, 'analytic drop chord');
assert.equal(results[6].events, 1);
direction(8, results[6].direction);
near(results[8].distance, chord, 0.002, 'density drop chord');
assert.equal(results[8].events, 1);
assert.equal(results[8].complete, 1);
for (const i of [0, 1, 2, 4, 6]) assert.equal(results[i].complete, 1);
for (let i = contactStart; i < contactStart + 5; i++) {
  direction(i, fixtures[i].d);
  assert.equal(results[i].events, 0, fixtures[i].name);
  assert.equal(results[i].complete, 1, fixtures[i].name);
  near(results[i].weight, 1, 1e-6, fixtures[i].name);
}
near(
  results[contactStart + 4].distance,
  0.14,
  1e-5,
  'wet floor includes only the artificial bottom gap',
);
for (const [offset, face] of [
  [5, [-1, 0, 0]],
  [6, [0, 1, 0]],
  [7, [-1, 0, 0]],
]) {
  const i = contactStart + offset;
  direction(i, refract(fixtures[i].d, face, 1.333));
  assert.equal(results[i].events, 1, fixtures[i].name);
  assert.ok(results[i].weight < 1, fixtures[i].name);
}
assert.deepEqual(errors, []);
console.log(
  'PASS: exit refraction, separated/thin layers, wedge, opaque termination, droplet chord and bounded TIR',
);
process.exit(0);
