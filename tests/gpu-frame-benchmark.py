"""Frozen production fluid/duck frame, GPU renderer ablations (macOS, not browser FPS).
Diagnostic variants change only this harness. Production rendering is untouched.
"""
from pathlib import Path
exec(compile(Path(__file__).with_name('gpu-duck.test.py').read_text().split('reset();history=[]')[0], __file__, 'exec'))
import statistics,struct
api('glGenQueries',None,I,c.POINTER(U));api('glBeginQuery',None,U,U);api('glEndQuery',None,U)
api('glGetQueryObjectui64v',None,U,U,c.POINTER(c.c_uint64));api('glDeleteQueries',None,I,c.POINTER(U))
reset()
for i in range(240):step(i/60)
grid('pos');reorder(False,count)
run('geometry','geometry',{'positions':'pos','sortedKeys':'keys','cellRanges':'ranges'})
run('volume','atlas',volume_inputs);filter_surface()
source='pos'
for size in [128,64,32,16,8,4,2,1]:
    name='bounds'+str(size);T[name]=target(size,max(1,size//2))
    run('bounds',name,{'source':source},{'firstLevel':int(source=='pos')});source=name
folder=root/'public/models/duck';meta=json.loads((folder/'mesh.json').read_text())
for name,file,height in [('bvh','bvh.bin',meta['bvhHeight']),('triangles','triangles.bin',meta['triangleHeight'])]:
    T[name]=target(256,height);raw=(folder/file).read_bytes();a=(F*(len(raw)//4)).from_buffer_copy(raw)
    G.glBindTexture(0x0DE1,T[name][0]);G.glTexSubImage2D(0x0DE1,0,0,0,256,height,0x1908,0x1406,a)
# Albedo sampling is retained; constant color replaces the image decode dependency.
T['albedo']=target(1,1);G.glBindTexture(0x0DE1,T['albedo'][0]);a=(F*4)(1,.75,.05,1)
G.glTexSubImage2D(0x0DE1,0,0,0,1,1,0x1908,0x1406,a)
vertex='''#version 300 es
precision highp float;out vec2 texcoord;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);texcoord=p;gl_Position=vec4(p*2.-1.,0.,1.);}
'''
base=sources['surfaceFragment']
def replace_once(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
no_reflection=replace_once(base,
 'vec3 reflected=reflect(rd,n);vec4 mirrorDuck=duckTrace(p+reflected*.006,reflected,1e5);\n  vec3 reflection=mirrorDuck.y>=0.?duckShade(mirrorDuck,reflected):sky(reflected);',
 'vec3 reflection=vec3(0.);')
no_thickness=replace_once(base,'float thickness=opticalPath(p+refracted*.003,refracted);','float thickness=.25;')
# Keep thickness ray truncation / duck intersection, remove only its density samples.
no_thickness_samples=replace_once(base,'for(int i=0;i<72;i++){','for(int i=0;i<0;i++){')
variants={
 'full':(base,{}),
 'reflection_toggle_off':(base,{'reflectionOn':0}),
 'reflection_removed':(no_reflection,{'reflectionOn':0}),
 'duck_hidden':(base,{'duckReady':0}),
 'thickness_samples_removed':(no_thickness_samples,{}),
 'thickness_path_removed':(no_thickness,{}),
 'water_hidden':(base,{'particleView':1}),
}
compiled={}
for name,(frag,_) in variants.items():
    if frag not in compiled:compiled[frag]=compile_program(vertex,frag)
    programs[name]=compiled[frag]
print('PASS: renderer variants compiled; frozen state at step 240',flush=True)
def norm(v):
    length=math.sqrt(sum(x*x for x in v));return [x/length for x in v]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
yaw=.58;pitch=.49;zoom=8
eye=[math.sin(yaw)*math.cos(pitch)*zoom,math.sin(pitch)*zoom,math.cos(yaw)*math.cos(pitch)*zoom]
forward=norm([-eye[0],-.05-eye[1],-eye[2]]);right=norm(cross(forward,[0,1,0]));up=cross(right,forward)
textures={'densityVolume':'surfaceTemp','waterBounds':'bounds1','duckState':'duck','duckBVH':'bvh','duckTriangles':'triangles','duckAlbedo':'albedo'}
def draw(name,w,h):
    t=T['screen'];p=programs[name];G.glBindFramebuffer(0x8D40,t[1]);G.glViewport(0,0,w,h);G.glUseProgram(p)
    G.glDisable(0x0BE2);G.glDisable(0x0B71)
    for unit,(uniform,tex) in enumerate(textures.items()):
        G.glActiveTexture(0x84C0+unit);G.glBindTexture(0x0DE1,T[tex][0]);G.glUniform1i(G.glGetUniformLocation(p,uniform.encode()),unit)
    values={'resolution':[w,h],'eye':eye,'cameraRight':right,'cameraUp':up,'cameraForward':forward,
      'volumeMin':[-2.08,-1.12,-1.56],'volumeMax':[2.08,4.08,1.56],'volumeSize':[128,160,96],
      'absorption':[1.25,.2,.065],'offsetX':.19,'time':4.,'lightPower':1.3,'isoDensity':1.15,
      'duckReady':1.,'reflectionOn':1.,'causticsOn':1.,'particleView':0.,'brushOn':0.,**variants[name][1]}
    for k,v in values.items():
        loc=G.glGetUniformLocation(p,k.encode())
        if isinstance(v,list):getattr(G,'glUniform%dfv'%len(v))(loc,1,(F*len(v))(*v))
        else:G.glUniform1f(loc,v)
    G.glDrawArrays(4,0,3)
    assert G.glGetError()==0

def measure(fn):
    q=U();G.glGenQueries(1,c.byref(q));G.glBeginQuery(0x88BF,q);fn();G.glEndQuery(0x88BF);G.glFinish()
    out=c.c_uint64();G.glGetQueryObjectui64v(q,0x8866,c.byref(out));G.glDeleteQueries(1,c.byref(q));return out.value/1e6
for w,h in [(1250,800),(813,520)]:
    T['screen']=target(w,h)
    # Match the onscreen color format, not a float32 output target.
    G.glBindTexture(0x0DE1,T['screen'][0]);G.glTexImage2D(0x0DE1,0,0x8058,w,h,0,0x1908,0x1401,None)
    samples={name:[] for name in variants}
    for frame in range(22):
        order=list(variants);order=order[frame%len(order):]+order[:frame%len(order)]
        for name in order:
            draw(name,w,h)
            # Time repeated draw commands after uniform/texture setup to reduce
            # Python/driver submission overhead in these short rendering passes.
            def batch():
                for _ in range(16):G.glDrawArrays(4,0,3)
            ms=measure(batch)/16
            if frame>=8:samples[name].append(ms)
    print('RENDER_ABLATION',json.dumps({'particles':quality,'resolution':[w,h],
      'gpu_ms_median':{k:round(statistics.median(v),3) for k,v in samples.items()}}),flush=True)

# Check that the full path really draws water: ablation must change framebuffer pixels.
draw('full',813,520);full=list(read('screen'))
draw('water_hidden',813,520);dry=list(read('screen'))
changed=sum(any(abs(full[i+c]-dry[i+c])>.02 for c in range(3)) for i in range(0,len(full),4))
assert changed>1000, ('Water path not exercised',changed)
print('PASS: water changes',changed,'pixels in the frozen render',flush=True)
