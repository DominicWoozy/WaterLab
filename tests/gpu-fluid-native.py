"""Numerical GPU regression tests on macOS OpenGL (no browser/UI automation).
ES 3.00 shaders are compiled as desktop 4.10 with only version/precision adaptation.
This checks the actual shader arithmetic, not WebGL browser compatibility or visual appearance.
"""
import ctypes as c
import json, math, re, subprocess, sys, time
from pathlib import Path
if sys.platform != 'darwin':
    print('SKIP: native GPU harness requires macOS'); sys.exit(0)
root = Path(__file__).resolve().parents[1]
sources = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', "import * as s from './app/gpu-fluid-shaders.ts'; import * as w from './app/water-shaders.ts'; console.log(JSON.stringify({...s,...w}));"], cwd=root))
# Test-only comparison switch; the app always includes solid boundary support.
if '--without-wall-support' in sys.argv:
    sources={k:re.sub(r'vec4 wallSupport\(vec3 p\)\{.*?\n\}', 'vec4 wallSupport(vec3 p){return vec4(0.);}', v, flags=re.S) for k,v in sources.items()}
# Compile the production traversal with only its reduction changed to count/moment.
probe_source=sources['lambdaFragment']
for before,after in [
    ('rho+=q*q; grad+=gradient; sum+=dot(gradient,gradient);','rho+=1.;grad+=d;'),
    ('vec4 wall=wallSupport(p)+duckSupport(p);float rho=wall.w,sum=0.;vec3 grad=wall.xyz;',
     'float rho=0.,sum=0.;vec3 grad=vec3(0.);'),
    ('result=vec4(-max(rho/REST-1.,0.)/(sum+dot(grad,grad)+2.),rho,0.,0.);','result=vec4(grad,rho);'),
]:
    assert probe_source.count(before)==1, 'Production shader changed: update the neighbor-only test probe'
    probe_source=probe_source.replace(before,after)
sources['neighborProbeFragment']=probe_source

G = c.CDLL('/System/Library/Frameworks/OpenGL.framework/OpenGL')
def api(name, result, *args):
    f = getattr(G, name); f.restype = result; f.argtypes = args; return f
I,U,F,P,S = c.c_int,c.c_uint,c.c_float,c.c_void_p,c.c_char_p
attrs=(I*4)(99,0x3200,73,0); pixel=P(); number=I(); context=P()
assert G.CGLChoosePixelFormat(attrs,c.byref(pixel),c.byref(number)) == 0, 'GPU unavailable; run with graphics access'
assert G.CGLCreateContext(pixel,None,c.byref(context)) == 0
assert G.CGLSetCurrentContext(context) == 0
api('glGetString',S,U)
print('GPU:',G.glGetString(0x1F01).decode(),flush=True)
for name,result,args in [
 ('glCreateShader',U,[U]),('glShaderSource',None,[U,I,c.POINTER(S),P]),('glCompileShader',None,[U]),('glGetShaderiv',None,[U,U,c.POINTER(I)]),('glGetShaderInfoLog',None,[U,I,P,P]),
 ('glCreateProgram',U,[]),('glAttachShader',None,[U,U]),('glLinkProgram',None,[U]),('glGetProgramiv',None,[U,U,c.POINTER(I)]),('glGetProgramInfoLog',None,[U,I,P,P]),('glUseProgram',None,[U]),
 ('glGenTextures',None,[I,c.POINTER(U)]),('glBindTexture',None,[U,U]),('glTexImage2D',None,[U,I,I,I,I,I,U,U,P]),('glTexParameteri',None,[U,U,I]),('glTexSubImage2D',None,[U,I,I,I,I,I,U,U,P]),
 ('glGenFramebuffers',None,[I,c.POINTER(U)]),('glBindFramebuffer',None,[U,U]),('glFramebufferTexture2D',None,[U,U,U,U,I]),('glCheckFramebufferStatus',U,[U]),
 ('glDrawBuffers',None,[I,c.POINTER(U)]),('glViewport',None,[I,I,I,I]),('glGenVertexArrays',None,[I,c.POINTER(U)]),('glBindVertexArray',None,[U]),('glDrawArrays',None,[U,I,I]),('glDrawArraysInstanced',None,[U,I,I,I]),
 ('glActiveTexture',None,[U]),('glGetUniformLocation',I,[U,S]),('glUniform1i',None,[I,I]),('glUniform1f',None,[I,F]),('glUniform2fv',None,[I,I,c.POINTER(F)]),('glUniform3fv',None,[I,I,c.POINTER(F)]),('glUniform4fv',None,[I,I,c.POINTER(F)]),
 ('glClearColor',None,[F,F,F,F]),('glClear',None,[U]),('glEnable',None,[U]),('glDisable',None,[U]),('glBlendFunc',None,[U,U]),('glReadPixels',None,[I,I,I,I,U,U,P]),('glGetError',U,[]),('glFinish',None,[])]: api(name,result,*args)
def compile_program(v,f):
    program=G.glCreateProgram()
    for typ,text in [(0x8B31,v),(0x8B30,f)]:
        text=text.replace('#version 300 es','#version 410 core')
        text=re.sub(r'precision\s+\w+\s+\w+\s*;','',text)
        text=re.sub(r'\b(highp|mediump|lowp)\b','',text)
        shader=G.glCreateShader(typ); source=S(text.encode()); G.glShaderSource(shader,1,c.byref(source),None);G.glCompileShader(shader)
        ok=I();G.glGetShaderiv(shader,0x8B81,c.byref(ok))
        if not ok.value:
            log=c.create_string_buffer(10000);G.glGetShaderInfoLog(shader,10000,None,log);raise AssertionError(log.value.decode())
        G.glAttachShader(program,shader)
    G.glLinkProgram(program);ok=I();G.glGetProgramiv(program,0x8B82,c.byref(ok))
    if not ok.value:
        log=c.create_string_buffer(10000);G.glGetProgramInfoLog(program,10000,None,log);raise AssertionError(log.value.decode())
    return program
programs={}
for name in ['radixRank','radixHistogram','radixScan','radixScatter','surfaceFilter','divergenceFactor','divergenceResidual','divergenceProject','neighborProbe','reorder','geometry','bounds','initialize','predict','key','sort','sortMerge','ranges','lambda','correct','velocity','viscosity','volume']:
    programs[name]=compile_program(sources['volumeVertex' if name=='volume' else 'radixScatterVertex' if name=='radixScatter' else 'computeVertex'],sources[name+'Fragment'])
compile_program(sources['fullscreenVertex'],sources['surfaceFragment'])
compile_program(sources['particleVertex'],sources['particleFragment'])
print('PASS: all compute, volume, surface and debug shaders compile/link',flush=True)
vao=U();G.glGenVertexArrays(1,c.byref(vao));G.glBindVertexArray(vao)
def target(w=256,h=128,half=False):
    t=U();fb=U();G.glGenTextures(1,c.byref(t));G.glBindTexture(0x0DE1,t)
    G.glTexImage2D(0x0DE1,0,0x822D if half else 0x8814,w,h,0,0x1903 if half else 0x1908,0x1406,None)
    for param in [0x2801,0x2800]:G.glTexParameteri(0x0DE1,param,0x2601 if half else 0x2600)
    for param in [0x2802,0x2803]:G.glTexParameteri(0x0DE1,param,0x812F)
    G.glGenFramebuffers(1,c.byref(fb));G.glBindFramebuffer(0x8D40,fb);G.glFramebufferTexture2D(0x8D40,0x8CE0,0x0DE1,t,0)
    assert G.glCheckFramebufferStatus(0x8D40)==0x8CD5
    G.glClearColor(0,0,0,0);G.glClear(0x4000)
    return (t.value,fb.value,w,h)
quality=int(sys.argv[1]) if len(sys.argv)>1 else 15000
count=quality
scale=(10000/quality)**(1/3)
sort_count=16384 if count<=16384 else 32768
T={name:target() for name in ['pos','pred','corr','vel','veltmp','keys','keytmp','lambda','geometry','metric0','metric1','metric2','sortpos','sortold','sortvel','factor']}
T['radixRanks']=target();T['radixHistogram']=target(256,16);T['radixTemp']=target(256,16)
T['ranges']=target(256,120);T['atlas']=target(1024,1920,True)
T['surfaceTemp']=target(1024,1920,True);T['surfaceFiltered']=target(1024,1920,True)
G.glBindFramebuffer(0x8D40,T['geometry'][1])
for attachment,name in enumerate(['metric0','metric1','metric2'],1):G.glFramebufferTexture2D(0x8D40,0x8CE0+attachment,0x0DE1,T[name][0],0)
G.glDrawBuffers(4,(U*4)(0x8CE0,0x8CE1,0x8CE2,0x8CE3))
volume_inputs={'positions':'geometry','metric0':'metric0','metric1':'metric1','metric2':'metric2'}
def run(name,out,inputs={},values={}):
    t=T[out];p=programs[name];G.glBindFramebuffer(0x8D40,t[1]);G.glViewport(0,0,t[2],t[3]);G.glUseProgram(p);G.glDisable(0x0BE2)
    G.glUniform1i(G.glGetUniformLocation(p,b'count'),count)
    G.glUniform1i(G.glGetUniformLocation(p,b'initialCount'),quality)
    G.glUniform1i(G.glGetUniformLocation(p,b'sortCount'),sort_count)
    G.glUniform1f(G.glGetUniformLocation(p,b'particleScale'),scale)
    if name in ['key','sort','sortMerge','radixRank','radixScatter']:G.glViewport(0,0,256,sort_count//256)
    elif name in ['radixHistogram','radixScan']:G.glViewport(0,0,256,sort_count//2048)
    elif name in ['predict','lambda','correct','velocity','viscosity','geometry','reorder','divergenceFactor','divergenceResidual','divergenceProject']:G.glViewport(0,0,256,max(1,math.ceil(count/256)))
    for unit,(uniform,tex) in enumerate(inputs.items()):
        assert T[tex]!=t
        G.glActiveTexture(0x84C0+unit);G.glBindTexture(0x0DE1,T[tex][0]);G.glUniform1i(G.glGetUniformLocation(p,uniform.encode()),unit)
    for name2,value in values.items():
        loc=G.glGetUniformLocation(p,name2.encode())
        if isinstance(value,(list,tuple)):getattr(G,'glUniform%dfv'%len(value))(loc,1,(F*len(value))(*value))
        elif name2 in ['stage','stride','previousCount','digitShift','scanStride']:G.glUniform1i(loc,int(value))
        else:G.glUniform1f(loc,value)
    if name=='volume':
        G.glClearColor(0,0,0,0);G.glClear(0x4000);G.glEnable(0x0BE2);G.glBlendFunc(1,1);G.glDrawArraysInstanced(4,0,6,count*20);G.glDisable(0x0BE2)
    elif name=='radixScatter':G.glDrawArrays(0,0,sort_count)
    else:G.glDrawArrays(4,0,3)
    error=G.glGetError();assert error==0,(name,hex(error))
def filter_surface():
    run('surfaceFilter','surfaceTemp',{'source':'atlas','guide':'atlas'},{'axis':[1,0,0]})
    run('surfaceFilter','surfaceFiltered',{'source':'surfaceTemp','guide':'atlas'},{'axis':[0,1,0]})
    run('surfaceFilter','surfaceTemp',{'source':'surfaceFiltered','guide':'atlas'},{'axis':[0,0,1]})

def read(name):
    t=T[name];G.glBindFramebuffer(0x8D40,t[1]);a=(F*(t[2]*t[3]*4))();G.glReadPixels(0,0,t[2],t[3],0x1908,0x1406,a);assert G.glGetError()==0;return a

def bitonic_grid(p):
    global sort_count
    sort_count=16384 if count<=16384 else 32768
    run('key','keys',{'positions':p})
    stage=2
    while stage<=sort_count:
        stride=stage//2
        while stride:
            run('sortMerge' if stride==4 else 'sort','keytmp',{'sortedKeys':'keys'},{'stage':stage,'stride':stride});T['keys'],T['keytmp']=T['keytmp'],T['keys'];stride=0 if stride==4 else stride//2
        stage*=2
    run('ranges','ranges',{'sortedKeys':'keys'})

def grid(p):
    global sort_count
    sort_count=16384 if count<=16384 else 32768
    run('key','keys',{'positions':p})
    for digit in [0,4,8,12]:
        run('radixRank','radixRanks',{'sortedKeys':'keys'},{'digitShift':digit})
        run('radixHistogram','radixHistogram',{'sortedKeys':'keys'},{'digitShift':digit})
        for stride in [1,16,256]:
            run('radixScan','radixTemp',{'radixHistogram':'radixHistogram'},{'scanStride':stride})
            T['radixHistogram'],T['radixTemp']=T['radixTemp'],T['radixHistogram']
        run('radixScatter','keytmp',{'radixRanks':'radixRanks','radixHistogram':'radixHistogram'})
        T['keys'],T['keytmp']=T['keytmp'],T['keys']
    run('ranges','ranges',{'sortedKeys':'keys'})

def reorder(predicted,previous):
    source='pred' if predicted else 'pos'
    G.glBindFramebuffer(0x8D40,T['sortpos'][1])
    for attachment,name in enumerate(['sortold','sortvel'],1):G.glFramebufferTexture2D(0x8D40,0x8CE0+attachment,0x0DE1,T[name][0],0)
    G.glDrawBuffers(3,(U*3)(0x8CE0,0x8CE1,0x8CE2))
    run('reorder','sortpos',{'positions':source,'oldPositions':'pos','velocities':'vel','sortedKeys':'keys'},{'previousCount':previous})
    for attachment in [1,2]:G.glFramebufferTexture2D(0x8D40,0x8CE0+attachment,0x0DE1,0,0)
    G.glDrawBuffers(1,(U*1)(0x8CE0))
    T[source],T['sortpos']=T['sortpos'],T[source]
    T['vel'],T['sortvel']=T['sortvel'],T['vel']

def step(t=0,gravity=9.8,splash=(0,0,0),previous=None,projection=True):
    run('predict','pred',{'positions':'pos','velocities':'vel'},{'dt':1/60,'time':t,'gravity':gravity,'agitation':0,'shake':0,'previousCount':count if previous is None else previous,'brush':[0,0,0,0],'brushVelocity':[0,0],'pourAt':[0,0],'splash':splash})
    grid('pred');reorder(True,count if previous is None else previous);ni={'sortedKeys':'keys','cellRanges':'ranges'}
    for _ in range(3):
        run('lambda','lambda',{'positions':'pred',**ni});run('correct','corr',{'positions':'pred','lambdas':'lambda',**ni});T['pred'],T['corr']=T['corr'],T['pred']
    run('velocity','veltmp',{'positions':'pred','oldPositions':'sortold'},{'dt':1/60,'previousCount':count if previous is None else previous})
    run('viscosity','vel',{'positions':'pred','velocities':'veltmp',**ni},{'viscosity':.025})
    if projection:
        run('divergenceFactor','factor',{'positions':'pred',**ni})
        for _ in range(2):
            run('divergenceResidual','lambda',{'positions':'pred','velocities':'vel','factors':'factor',**ni})
            run('divergenceProject','veltmp',{'positions':'pred','velocities':'vel','lambdas':'lambda',**ni});T['vel'],T['veltmp']=T['veltmp'],T['vel']
    T['pos'],T['pred']=T['pred'],T['pos']
run('initialize','pos');initial=read('pos');assert sum(initial[4*i+3]>.5 for i in range(32768))==quality
assert len({tuple(initial[4*i:4*i+3]) for i in range(count)})==quality
print('PASS:',quality,'unique initialized GPU particles',flush=True)
grid('pos');keys=read('keys');pairs=[tuple(keys[4*i:4*i+2]) for i in range(sort_count)];assert pairs==sorted(pairs)
assert set(int(x[1]) for x in pairs[:count])==set(range(count))
ranges=read('ranges')
for k in range(32*40*24):
    lo,hi=map(int,ranges[4*k:4*k+2]);assert 0<=lo<=hi<=count
    if lo<hi:assert all(pairs[j][0]==k for j in range(lo,hi))
    if lo>0:assert pairs[lo-1][0]<k
    assert pairs[hi][0]>k
print('PASS: complete GPU sort and every cell range',flush=True)
started=time.monotonic()
for i in range(120):step(i/60)
positions=read('pos');vel=read('vel')
assert all(math.isfinite(x) for x in positions)
assert all(-1.7801<=positions[4*i]<=1.7801 and -.9171<=positions[4*i+1]<=3.8001 and -1.2801<=positions[4*i+2]<=1.2801 for i in range(count))
energy=sum(sum(vel[4*i+j]**2 for j in range(3)) for i in range(count))/count
assert energy<1.,energy
print('PASS: 120 steps, calm mean squared speed %.4f; elapsed %.2fs'%(energy,time.monotonic()-started),flush=True)
step(2,splash=(0,0,3));v=read('vel');assert max(v[4*i+1] for i in range(count))>1
for i in range(30):step(2+i/60)
for i in range(112):
    previous=count;count=min(30000,quality+2000,count+18);step(3+i/60,previous=previous)
p=read('pos');
if quality<30000:assert sum(p[4*i+1]>1.7 for i in range(count))>=2
assert all(math.isfinite(x) for x in p)
print('PASS: splash response and capacity/injection', count,flush=True)
# Compare actual half-float additive volume against the CPU kernel at grid nodes.
grid('pos');reorder(False,count);run('geometry','geometry',{'positions':'pos','sortedKeys':'keys','cellRanges':'ranges'})
run('volume','atlas',volume_inputs);atlas=read('atlas');geo=read('geometry');metrics=[read('metric'+str(i)) for i in range(3)]
def density_node(x,y,z):return atlas[4*((z//8*160+y)*1024+z%8*128+x)]
for x,y,z in [(64,11,48),(64,92,48),(33,11,40)]:
    world=[-2.08+x/127*4.16,-1.12+y/159*5.2,-1.56+z/95*3.12];expected=0
    for i in range(count):
        blend=max(0,min(1,(geo[4*i+3]-.15)/1.05));blend=blend*blend*(3-2*blend)
        radius=max(.095,(.1+(.19-.1)*blend)*scale)
        d=[(world[j]-geo[4*i+j])/radius for j in range(3)]
        r2=sum(d[row]*metrics[col][4*i+row]*d[col] for row in range(3) for col in range(3))
        if r2<1:expected+=(1-r2)**3*(1+max(0,1-geo[4*i+3])*.8)
    actual=density_node(x,y,z);assert abs(actual-expected)<max(.02,expected*.025),(actual,expected)
assert max(atlas[::4])>1.15
source='pos'
for size in [128,64,32,16,8,4,2,1]:
    name='bounds'+str(size);T[name]=target(size,max(1,size//2))
    run('bounds',name,{'source':source},{'firstLevel':int(source=='pos')});source=name
assert abs(read('bounds1')[1]-max(p[4*i+1] for i in range(count)))<.00001
print('PASS: GPU maximum-height reduction matches particle state',flush=True)
print('PASS: GPU density atlas agrees with CPU kernel (half-float tolerance)',flush=True)
# Isolated drops must shrink in world space and remain visible at off-grid positions.
count=1
for offset in [0., .25, .5, .75]:
    pos=(F*(256*128*4))(); pressure=(F*(256*128*4))()
    px=-2.08+(64+offset)/127*4.16
    py=-1.12+(50+offset)/159*5.2
    pz=-1.56+(48+offset)/95*3.12
    pos[0:4]=[px,py,pz,1.]
    for name,data in [('pos',pos),('lambda',pressure)]:
        G.glBindTexture(0x0DE1,T[name][0]);G.glTexSubImage2D(0x0DE1,0,0,0,256,128,0x1908,0x1406,data)
    grid('pos');reorder(False,count);run('geometry','geometry',{'positions':'pos','sortedKeys':'keys','cellRanges':'ranges'})
    run('volume','atlas',volume_inputs);atlas=read('atlas')
    filter_surface();filtered=read('surfaceTemp')
    assert all(atlas[j]==filtered[j] for j in range(0,len(atlas),4)), 'isolated spray was altered by bulk filter'
    wet=[]
    for z in range(43,54):
        for y in range(45,56):
            for x in range(59,70):
                if density_node(x,y,z)>1.15:
                    wet.append((-2.08+x/127*4.16,-1.12+y/159*5.2,-1.56+z/95*3.12))
    assert wet, ('off-grid droplet disappeared',offset)
    assert max(math.dist((px,py,pz),point) for point in wet)<.04
print('PASS: small isolated spray remains visible at four sub-voxel offsets, radius < 0.04',flush=True)
# Surface regression: identical jittered slab, isotropic vs covariance reconstruction.
count=quality
jitter=(F*(256*128*4))(*initial)
for i in range(count):jitter[4*i+1]+=.012*math.sin(i*17.3)
def upload(name,data):
    G.glBindTexture(0x0DE1,T[name][0]);G.glTexSubImage2D(0x0DE1,0,0,0,256,128,0x1908,0x1406,data)
upload('pos',jitter);grid('pos');reorder(False,count);run('geometry','geometry',{'positions':'pos','sortedKeys':'keys','cellRanges':'ranges'})
shape=read('geometry');tensor=[read('metric'+str(i)) for i in range(3)]
for i in range(count):
    m=[[tensor[col][4*i+row] for col in range(3)] for row in range(3)]
    det=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
    assert abs(det-1)<.002 and m[0][0]>0 and m[0][0]*m[1][1]-m[0][1]*m[1][0]>0
run('volume','atlas',volume_inputs);atlas=read('atlas')
def field(x,y,z):
    u=(x+2.08)/4.16*127;v=(y+1.12)/5.2*159;w=(z+1.56)/3.12*95
    ix,iy,iz=math.floor(u),math.floor(v),math.floor(w);fx,fy,fz=u-ix,v-iy,w-iz
    return sum(density_node(ix+dx,iy+dy,iz+dz)*(fx if dx else 1-fx)*(fy if dy else 1-fy)*(fz if dz else 1-fz) for dx in range(2) for dy in range(2) for dz in range(2))
def surface_rms():
    heights=[]
    for ix in range(11):
        for iz in range(9):
            x=-.9+ix*.18;z=-.6+iz*.15;hi=-.2;lo=hi
            for _ in range(100):
                lo-=.008
                if field(x,lo,z)>1.15:break
                hi=lo
            assert hi>-.9
            for _ in range(8):
                mid=(lo+hi)/2
                if field(x,mid,z)>1.15:lo=mid
                else:hi=mid
            heights.append((lo+hi)/2)
    mean=sum(heights)/len(heights)
    return math.sqrt(sum((h-mean)**2 for h in heights)/len(heights))
anisotropic=surface_rms()
jitter=(F*(256*128*4))(*read('pos'))
for i in range(count):jitter[4*i+3]=shape[4*i+3]
upload('geometry',jitter)
for col in range(3):
    identity=(F*(256*128*4))()
    for i in range(count):identity[4*i+col]=1;identity[4*i+3]=1
    upload('metric'+str(col),identity)
run('volume','atlas',volume_inputs);atlas=read('atlas');isotropic=surface_rms()
print('Surface RMS: isotropic %.3f mm, covariance %.3f mm'%(isotropic*1000,anisotropic*1000),flush=True)
# Require improvement when noise is measurable; allow a 0.2 mm floor for already-flat surfaces.
assert anisotropic<max(.0002,isotropic*.9),(anisotropic,isotropic)
print('PASS: positive definite volume-preserving tensors and bounded surface noise',flush=True)
count=0;run('volume','atlas',volume_inputs);assert max(read('atlas')[::4])==0
filter_surface();assert max(read('surfaceTemp')[::4])==0
print('PASS: empty volume clears without stale water',flush=True)
# Reorder permutation preserves correlated state; newborn tags survive arbitrary permutation.
import random
rng=random.Random(71);count=400;previous=380
state=(F*(256*128*4))();velocity=(F*(256*128*4))()
for i in range(count):
    state[4*i:4*i+4]=[rng.uniform(-.5,.5),rng.uniform(-.8,.2),rng.uniform(-.5,.5),1]
    velocity[4*i:4*i+4]=[i*.001,-i*.002,i*.003,float(i)]
upload('pos',state);upload('vel',velocity);grid('pos');indices=read('keys');reorder(False,previous)
ordered=read('pos');ordered_vel=read('vel');old=read('sortold')
for i in range(count):
    j=int(indices[4*i+1]);assert list(ordered[4*i:4*i+4])==list(state[4*j:4*j+4])
    assert list(ordered_vel[4*i:4*i+4])==list(velocity[4*j:4*j+4])
    assert old[4*i+3]==int(j<previous)
print('PASS: state permutation and newborn tags remain aligned',flush=True)
# Move every neighbour up to the full correction budget without rebuilding the grid.
for i in range(count):
    d=[rng.uniform(-1,1) for _ in range(3)];length=math.sqrt(sum(v*v for v in d))
    for axis in range(3):ordered[4*i+axis]+=d[axis]/length*.051*scale
upload('pos',ordered);run('neighborProbe','lambda',{'positions':'pos','cellRanges':'ranges'})
probe=read('lambda');hits=0
for i in range(count):
    total=[0.,0.,0.];n=0
    for j in range(count):
        if i==j:continue
        d=[ordered[4*i+k]-ordered[4*j+k] for k in range(3)];r2=sum(v*v for v in d)
        if 1e-12<r2<(.17*scale)**2:
            n+=1
            for k in range(3):total[k]+=d[k]
    assert probe[4*i+3]==n,(i,probe[4*i+3],n)
    assert all(abs(probe[4*i+k]-total[k])<1e-5 for k in range(3))
    hits+=n
assert hits>200
print('PASS: culled direct traversal matches brute force after maximum corrections',hits,'neighbours',flush=True)
# Projection should remove compression, not indiscriminately damp moving water.
count=quality;state=(F*(256*128*4))(*initial)
for i in range(count):
    state[i*4+1]+=.6;state[i*4]*=.7;state[i*4+2]*=.7
upload('pos',state);grid('pos');reorder(False,count);state=read('pos')
ni={'positions':'pos','cellRanges':'ranges'}
run('divergenceFactor','factor',ni)
for flow in ['rigid','compressing']:
    velocity=(F*(256*128*4))()
    for i in range(count):
        x,y,z=state[i*4:i*4+3]
        velocity[i*4:i*4+3]=[1.1-z,.2,-.4+x] if flow=='rigid' else [-x*.5,-(y+.15)*.5,-z*.5]
    upload('vel',velocity)
    run('divergenceResidual','lambda',{**ni,'velocities':'vel','factors':'factor'})
    before=read('lambda');before_rate=sum(max(0,before[i*4+1]) for i in range(count))
    for _ in range(2):
        run('divergenceResidual','lambda',{**ni,'velocities':'vel','factors':'factor'})
        run('divergenceProject','veltmp',{**ni,'velocities':'vel','lambdas':'lambda'});T['vel'],T['veltmp']=T['veltmp'],T['vel']
    projected=read('vel')
    if flow=='rigid':assert max(abs(projected[i]-velocity[i]) for i in range(count*4))<1e-5
    else:
        run('divergenceResidual','lambda',{**ni,'velocities':'vel','factors':'factor'})
        residual=read('lambda');after_rate=sum(max(0,residual[i*4+1]) for i in range(count))
        assert after_rate<before_rate*.95,(after_rate,before_rate)
        for axis in range(3):assert abs(sum(projected[i*4+axis]-velocity[i*4+axis] for i in range(count))/count)<1e-6
        print('PASS: compression projection reduces positive density rate %.1f%% and preserves momentum'%((1-after_rate/before_rate)*100),flush=True)
print('PASS: uniform translation plus rigid rotation is preserved by projection',flush=True)
print('All native GPU checks passed.',flush=True)
