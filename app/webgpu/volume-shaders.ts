import { common, neighbors } from './common.ts';
import { detailCommon } from './detail-shaders.ts';
export const shapeCommon = /* wgsl */ `
struct Shape {center:vec4f,m0:vec4f,m1:vec4f,m2:vec4f}
const VMIN=vec3f(-2.08,-1.12,-1.56);const VMAX=vec3f(2.08,4.08,1.56);
const VSIZE=vec3u(128,160,96);
const IDENTITY=mat3x3f(vec3f(1.,0.,0.),vec3f(0.,1.,0.),vec3f(0.,0.,1.));
fn determinant3(a:mat3x3f)->f32{return dot(a[0],cross(a[1],a[2]));}
fn inverse3(a:mat3x3f)->mat3x3f{return transpose(mat3x3f(cross(a[1],a[2]),cross(a[2],a[0]),cross(a[0],a[1])))*(1./determinant3(a));}
fn voxelIndex(p:vec3u)->u32{return p.x+128u*(p.y+160u*p.z);}
`;
export const volumeShaders: Record<string, string> = {
  geometry:
    common +
    shapeCommon +
    detailCommon +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read> input:array<Particle>;
 @group(0) @binding(2) var<storage,read_write> shapes:array<Shape>;
 @group(0) @binding(3) var<storage,read> starts:array<u32>;
 @group(0) @binding(4) var<storage,read_write> bounds:array<atomic<u32>>;
 @group(0) @binding(5) var<storage,read_write> details:array<Detail>;
 @group(0) @binding(6) var<storage,read_write> draw:array<atomic<u32>>;
 @group(0) @binding(7) var<uniform> settings:vec4u;
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=P.counts.x){return;}let p=input[i].pos.xyz;
 var total=1.;var rho=0.;var nearby=0.;var mean=vec3f(0.);var cov=IDENTITY*0.;
 ${neighbors('let w=q*q*q;total+=w;mean-=diff*w;cov+=mat3x3f(diff*diff.x,diff*diff.y,diff*diff.z)*w;rho+=q*q;nearby+=1.;')}
 mean/=total;cov=cov*(1./total)-mat3x3f(mean*mean.x,mean*mean.y,mean*mean.z);
 let tr=max(cov[0][0]+cov[1][1]+cov[2][2],1e-8);
 var kind=0.;var sheetNormal=vec3f(0.,1.,0.);
 if(settings.x!=0u){
  if(i==0u){atomicStore(&draw[0],6u);}
  if(rho<.18&&nearby<=4.){kind=1.;}
  if(nearby>=3.&&rho<2.4){
   var n=cross(cov[0],cov[1]);let n1=cross(cov[1],cov[2]);let n2=cross(cov[2],cov[0]);
   if(dot(n1,n1)>dot(n,n)){n=n1;}if(dot(n2,n2)>dot(n,n)){n=n2;}
   let square=dot(cov[0],cov[0])+dot(cov[1],cov[1])+dot(cov[2],cov[2]);
   let spread=(tr*tr-square)*.5/(tr*tr);
   if(dot(n,n)>1e-22){n=normalize(n);if(dot(n,cov*n)/tr<.025&&spread>.14){kind=2.;sheetNormal=n;}}
  }
 }
 if(kind>0.){
  let radius=h()*pow(.1/REST,1./3.);var metric=IDENTITY*(1./(radius*radius));var boundRadius=radius;
  if(kind>1.5){
   let tangent=h()*.78;let thin=radius*radius*radius/(tangent*tangent);let outer=mat3x3f(sheetNormal*sheetNormal.x,sheetNormal*sheetNormal.y,sheetNormal*sheetNormal.z);
   metric=IDENTITY*(1./(tangent*tangent))+outer*(1./(thin*thin)-1./(tangent*tangent));boundRadius=tangent;
  }
  let slot=atomicAdd(&draw[1],1u);
  details[slot]=Detail(vec4f(p,kind),vec4f(metric[0],boundRadius),vec4f(metric[1],f32(i)),vec4f(metric[2],particleMass()));
 }
 cov+=IDENTITY*(tr*.18+1e-7);
 cov=cov*(1./pow(max(determinant3(cov),1e-24),1./3.));
 let confidence=smoothstep(5.,14.,nearby)*smoothstep(.35,1.5,rho);
 cov=IDENTITY*(1.-confidence)+cov*confidence;cov=cov*(1./pow(max(determinant3(cov),1e-8),1./3.));
 let metric=inverse3(cov);let centre=p+limited(mean*.55*confidence,.025*P.clock.z);
 shapes[i]=Shape(vec4f(centre,rho),vec4f(metric[0],sqrt(cov[0][0])),vec4f(metric[1],sqrt(cov[1][1])),vec4f(metric[2],select(sqrt(cov[2][2]),-kind,kind>0.)));
 if(kind==0.){atomicMax(&bounds[0],bitcast<u32>(p.y+2.));}
 }`,
  density:
    common +
    shapeCommon +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read> shapes:array<Shape>;
 @group(0) @binding(2) var<storage,read> starts:array<u32>;
 @group(0) @binding(3) var<storage,read_write> field:array<f32>;
 @group(0) @binding(4) var<storage,read> bounds:array<u32>;
 @compute @workgroup_size(8,4,1) fn main(@builtin(global_invocation_id) gid:vec3u){
 if(any(gid>=VSIZE)){return;}let index=voxelIndex(gid);
 let p=VMIN+vec3f(gid)/vec3f(VSIZE-vec3u(1))*(VMAX-VMIN);
 if(P.counts.x==0u||p.y>bitcast<f32>(bounds[0])-2.+.36||p.y<-.97||abs(p.x)>1.9||abs(p.z)>1.4){field[index]=0.;return;}
 let base=cell(p);var value=0.;
 // Covariance regularization bounds max kernel stretch by cbrt(1.18/.18)<1.873.
 // Include render-centre displacement; the FINAL-position grid is rebuilt first.
 let reach=(.19*1.873+.025)*P.clock.z;let cellSize=.225*P.clock.z;
 for(var z=max(0,base.z-2);z<=min(23,base.z+2);z++){
 for(var y=max(0,base.y-2);y<=min(39,base.y+2);y++){
  let lo=vec2f(-1.19,-1.53)+vec2f(f32(y),f32(z))*cellSize;
  let gap=max(max(lo-p.yz,p.yz-lo-vec2f(cellSize)),vec2f(0.));
  let remaining=reach*reach-dot(gap,gap);if(remaining<0.){continue;}
  let extent=sqrt(remaining);
  let x0=clamp(i32(floor((p.x-extent+2.04)/cellSize)),0,31);
  let x1=clamp(i32(floor((p.x+extent+2.04)/cellSize)),0,31);
  let row=32*(y+40*z);let begin=starts[u32(row+x0)];let end=starts[u32(row+x1+1)];
  for(var j=begin;j<end;j++){
   if(shapes[j].m2.w<0.){continue;}
   let centre=shapes[j].center;let radius=max(.095,mix(.1,.19,smoothstep(.15,1.2,centre.w))*P.clock.z);
   let diff=(p-centre.xyz)/radius;if(dot(diff,diff)>3.51){continue;}
   let metric=mat3x3f(shapes[j].m0.xyz,shapes[j].m1.xyz,shapes[j].m2.xyz);
   let r2=dot(diff,metric*diff);if(r2<1.){let q=1.-r2;value+=q*q*q*(1.+max(0.,1.-centre.w)*.8);}
  }
 }}
 field[index]=value;
 }`,
};
for (let axis = 0; axis < 3; axis++) {
  volumeShaders[`filter${axis}`] =
    common +
    shapeCommon +
    /* wgsl */ `
 @group(0) @binding(1) var<storage,read> input:array<f32>;
 @group(0) @binding(2) var<storage,read> guide:array<f32>;
 @group(0) @binding(3) var<storage,read_write> output:array<f32>;
 fn at(p:vec3i)->u32{return voxelIndex(vec3u(clamp(p,vec3i(0),vec3i(VSIZE)-vec3i(1))));}
 @compute @workgroup_size(8,4,4) fn main(@builtin(global_invocation_id) gid:vec3u){
 if(any(gid>=VSIZE)){return;}let index=voxelIndex(gid);let p=vec3i(gid);let centre=guide[index];
 if(centre<.02){output[index]=centre;return;}
 var direction=vec3i(0);direction[${axis}]=1;var sum=0.;var peak=0.;
 for(var k=-2;k<=2;k++){let j=at(p+direction*k);var w=1.;if(abs(k)==1){w=4.;}if(k==0){w=6.;}sum+=input[j]*w;peak=max(peak,guide[j]);}
 for(var a=0;a<3;a++){var delta=vec3i(0);delta[a]=2;peak=max(peak,max(guide[at(p+delta)],guide[at(p-delta)]));}
 let gradient=vec3f(guide[at(p+vec3i(1,0,0))]-guide[at(p-vec3i(1,0,0))],guide[at(p+vec3i(0,1,0))]-guide[at(p-vec3i(0,1,0))],guide[at(p+vec3i(0,0,1))]-guide[at(p-vec3i(0,0,1))]);
 let nw=gradient[${axis}]*gradient[${axis}]/max(dot(gradient,gradient),1e-8);
 let blend=smoothstep(1.85,4.,peak)*.85*(1.-.95*nw);output[index]=mix(centre,sum/16.,blend);
 }`;
}
