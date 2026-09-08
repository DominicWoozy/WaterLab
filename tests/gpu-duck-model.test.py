"""Compare the imported-mesh GPU BVH against independent brute-force triangle rays."""
from pathlib import Path
prefix=Path(__file__).with_name('gpu-fluid-native.py').read_text().split("run('initialize','pos');initial=")[0]
exec(compile(prefix,__file__,'exec'))
import struct, random
folder=root/'public/models/duck';meta=json.loads((folder/'mesh.json').read_text())
def upload(name,w,h,data):
 T[name]=target(w,h);G.glBindTexture(0x0DE1,T[name][0]);a=(F*len(data))(*data);G.glTexSubImage2D(0x0DE1,0,0,0,w,h,0x1908,0x1406,a)
for name,file,height in [('bvh','bvh.bin',meta['bvhHeight']),('triangles','triangles.bin',meta['triangleHeight'])]:
 raw=(folder/file).read_bytes();values=struct.unpack('<'+'f'*(len(raw)//4),raw);upload(name,256,height,values)
 if name=='triangles':triangles=[values[i*24:i*24+24] for i in range(meta['triangles'])]
state=[0.,0.,0.,1.,0.,0.,0.,1.]+[0.]*8;upload('duck',4,1,state)
rng=random.Random(812);origins=[];directions=[]
for i in range(128):
 angle=i/128*math.tau;o=[1.8*math.sin(angle),.3+1.1*math.sin(angle*3),1.8*math.cos(angle)]
 aim=[rng.uniform(-.31,.43),rng.uniform(-.14,.55),rng.uniform(-.26,.26)];d=[aim[a]-o[a] for a in range(3)];length=math.sqrt(sum(v*v for v in d));d=[v/length for v in d];origins.append(o);directions.append(d)
upload('origins',128,1,[v for p in origins for v in p+[0.]])
upload('directions',128,1,[v for p in directions for v in p+[0.]])
duck_source=json.loads(subprocess.check_output(['node','--input-type=module','-e',"import {duckRender} from './app/duck-shaders.ts';console.log(JSON.stringify(duckRender))"],cwd=root))
shader='#version 300 es\nprecision highp float;precision highp int;uniform float lightPower;vec3 sky(vec3 r){return vec3(1.);}\n'+duck_source+'''\nuniform highp sampler2D origins,directions;out vec4 result;
void main(){ivec2 at=ivec2(gl_FragCoord.xy);result=duckTrace(texelFetch(origins,at,0).xyz,texelFetch(directions,at,0).xyz,100.);}
'''
programs['meshProbe']=compile_program(sources['computeVertex'],shader);T['probe']=target(128,1)
run('meshProbe','probe',{'duckState':'duck','duckBVH':'bvh','duckTriangles':'triangles','origins':'origins','directions':'directions'},{'duckReady':1})
actual=read('probe')
def sub(a,b):return [x-y for x,y in zip(a,b)]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
hits=0
for i,(o,d) in enumerate(zip(origins,directions)):
 best=100
 for tri in triangles:
  a,b,z=tri[:3],tri[4:7],tri[8:11];e1,e2=sub(b,a),sub(z,a);h=cross(d,e2);det=dot(e1,h)
  if abs(det)<1e-9:continue
  s=sub(o,a);u=dot(s,h)/det
  if u<0 or u>1:continue
  v=cross(s,e1);w=dot(d,v)/det
  if w<0 or u+w>1:continue
  t=dot(e2,v)/det
  if .0004<t<best:best=t
 assert abs(actual[i*4]-best)<2e-4,(i,actual[i*4],best)
 if best<100:hits+=1
assert hits>40
print('PASS: 128 GPU rays match all 4,212 original triangles, including misses;',hits,'hits',flush=True)
# Asset integrity, original normals/UVs and balanced tree bounds are independent of rendering.
assert len(triangles)==4212 and meta['vertices']==2399
assert all(math.isfinite(v) for t in triangles for v in t)
assert all(abs(sum(t[j+k]**2 for k in range(3))-1)<.002 for t in triangles for j in [12,16,20])
print('PASS: imported model topology, original smooth normals and finite packed data',flush=True)
