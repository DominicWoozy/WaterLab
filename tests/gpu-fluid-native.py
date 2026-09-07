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
 ('glViewport',None,[I,I,I,I]),('glGenVertexArrays',None,[I,c.POINTER(U)]),('glBindVertexArray',None,[U]),('glDrawArrays',None,[U,I,I]),('glDrawArraysInstanced',None,[U,I,I,I]),
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
for name in ['bounds','initialize','predict','key','sort','ranges','lambda','correct','velocity','viscosity','volume']:
    programs[name]=compile_program(sources['volumeVertex' if name=='volume' else 'computeVertex'],sources[name+'Fragment'])
compile_program(sources['fullscreenVertex'],sources['surfaceFragment'])
compile_program(sources['particleVertex'],sources['particleFragment'])
print('PASS: all compute, volume, surface and debug shaders compile/link',flush=True)
vao=U();G.glGenVertexArrays(1,c.byref(vao));G.glBindVertexArray(vao)
def target(w=128,h=128,half=False):
    t=U();fb=U();G.glGenTextures(1,c.byref(t));G.glBindTexture(0x0DE1,t)
    G.glTexImage2D(0x0DE1,0,0x822D if half else 0x8814,w,h,0,0x1903 if half else 0x1908,0x1406,None)
    for param in [0x2801,0x2800]:G.glTexParameteri(0x0DE1,param,0x2601 if half else 0x2600)
    for param in [0x2802,0x2803]:G.glTexParameteri(0x0DE1,param,0x812F)
    G.glGenFramebuffers(1,c.byref(fb));G.glBindFramebuffer(0x8D40,fb);G.glFramebufferTexture2D(0x8D40,0x8CE0,0x0DE1,t,0)
    assert G.glCheckFramebufferStatus(0x8D40)==0x8CD5
    G.glClearColor(0,0,0,0);G.glClear(0x4000)
    return (t.value,fb.value,w,h)
count=10000
T={name:target() for name in ['pos','pred','corr','vel','veltmp','keys','keytmp','lambda']}
T['ranges']=target(128,137);T['atlas']=target(1024,1920,True)
def run(name,out,inputs={},values={}):
    t=T[out];p=programs[name];G.glBindFramebuffer(0x8D40,t[1]);G.glViewport(0,0,t[2],t[3]);G.glUseProgram(p);G.glDisable(0x0BE2)
    G.glUniform1i(G.glGetUniformLocation(p,b'count'),count)
    for unit,(uniform,tex) in enumerate(inputs.items()):
        assert T[tex]!=t
        G.glActiveTexture(0x84C0+unit);G.glBindTexture(0x0DE1,T[tex][0]);G.glUniform1i(G.glGetUniformLocation(p,uniform.encode()),unit)
    for name2,value in values.items():
        loc=G.glGetUniformLocation(p,name2.encode())
        if isinstance(value,(list,tuple)):getattr(G,'glUniform%dfv'%len(value))(loc,1,(F*len(value))(*value))
        elif name2 in ['stage','stride','previousCount']:G.glUniform1i(loc,int(value))
        else:G.glUniform1f(loc,value)
    if name=='volume':
        G.glClearColor(0,0,0,0);G.glClear(0x4000);G.glEnable(0x0BE2);G.glBlendFunc(1,1);G.glDrawArraysInstanced(4,0,6,count*12);G.glDisable(0x0BE2)
    else:G.glDrawArrays(4,0,3)
    error=G.glGetError();assert error==0,(name,hex(error))
def read(name):
    t=T[name];G.glBindFramebuffer(0x8D40,t[1]);a=(F*(t[2]*t[3]*4))();G.glReadPixels(0,0,t[2],t[3],0x1908,0x1406,a);assert G.glGetError()==0;return a

def grid(p):
    run('key','keys',{'positions':p})
    stage=2
    while stage<=16384:
        stride=stage//2
        while stride:
            run('sort','keytmp',{'sortedKeys':'keys'},{'stage':stage,'stride':stride});T['keys'],T['keytmp']=T['keytmp'],T['keys'];stride//=2
        stage*=2
    run('ranges','ranges',{'sortedKeys':'keys'})

def step(t=0,gravity=9.8,splash=(0,0,0),previous=None):
    run('predict','pred',{'positions':'pos','velocities':'vel'},{'dt':1/60,'time':t,'gravity':gravity,'agitation':0,'shake':0,'previousCount':count if previous is None else previous,'brush':[0,0,0,0],'brushVelocity':[0,0],'pourAt':[0,0],'splash':splash})
    grid('pred');ni={'sortedKeys':'keys','cellRanges':'ranges'}
    for _ in range(2):
        run('lambda','lambda',{'positions':'pred',**ni});run('correct','corr',{'positions':'pred','lambdas':'lambda',**ni});T['pred'],T['corr']=T['corr'],T['pred']
    run('velocity','veltmp',{'positions':'pred','oldPositions':'pos'},{'dt':1/60,'previousCount':count if previous is None else previous})
    run('viscosity','vel',{'positions':'pred','velocities':'veltmp',**ni},{'viscosity':.025});T['pos'],T['pred']=T['pred'],T['pos']
run('initialize','pos');initial=read('pos');assert sum(initial[4*i+3]>.5 for i in range(16384))==10000
assert len({tuple(initial[4*i:4*i+3]) for i in range(count)})==10000
print('PASS: 10,000 unique initialized GPU particles',flush=True)
grid('pos');keys=read('keys');pairs=[tuple(keys[4*i:4*i+2]) for i in range(16384)];assert pairs==sorted(pairs)
assert set(int(x[1]) for x in pairs[:count])==set(range(count))
ranges=read('ranges')
for k in range(26*32*21):
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
    previous=count;count=min(12000,count+18);step(3+i/60,previous=previous)
p=read('pos');assert all(p[4*i+1]>1.7 for i in range(11998,12000))
assert all(math.isfinite(x) for x in p)
print('PASS: splash response and GPU particle injection to 12,000',flush=True)
# Compare actual half-float additive volume against the CPU kernel at grid nodes.
run('volume','atlas',{'positions':'pos','lambdas':'lambda'});atlas=read('atlas');lam=read('lambda')
def density_node(x,y,z):return atlas[4*((z//8*160+y)*1024+z%8*128+x)]
for x,y,z in [(64,11,48),(64,92,48),(33,11,40)]:
    world=[-2.08+x/127*4.16,-1.12+y/159*5.2,-1.56+z/95*3.12];expected=0
    for i in range(count):
        blend=max(0,min(1,(lam[4*i+1]-.15)/1.05));blend=blend*blend*(3-2*blend)
        radius=.1+(.19-.1)*blend
        r2=sum(((world[j]-p[4*i+j])/radius)**2 for j in range(3))
        if r2<1:expected+=(1-r2)**3*(1+max(0,1-lam[4*i+1])*.8)
    actual=density_node(x,y,z);assert abs(actual-expected)<max(.02,expected*.025),(actual,expected)
assert max(atlas[::4])>1.15
source='pos'
for size in [64,32,16,8,4,2,1]:
    name='bounds'+str(size);T[name]=target(size,size)
    run('bounds',name,{'source':source},{'firstLevel':int(source=='pos')});source=name
assert abs(read('bounds1')[1]-max(p[4*i+1] for i in range(count)))<.00001
print('PASS: GPU maximum-height reduction matches particle state',flush=True)
print('PASS: GPU density atlas agrees with CPU kernel (half-float tolerance)',flush=True)
# Isolated drops must shrink in world space and remain visible at off-grid positions.
count=1
for offset in [0., .25, .5, .75]:
    pos=(F*(128*128*4))(); pressure=(F*(128*128*4))()
    px=-2.08+(64+offset)/127*4.16
    py=-1.12+(50+offset)/159*5.2
    pz=-1.56+(48+offset)/95*3.12
    pos[0:4]=[px,py,pz,1.]
    for name,data in [('pos',pos),('lambda',pressure)]:
        G.glBindTexture(0x0DE1,T[name][0]);G.glTexSubImage2D(0x0DE1,0,0,0,128,128,0x1908,0x1406,data)
    run('volume','atlas',{'positions':'pos','lambdas':'lambda'});atlas=read('atlas')
    wet=[]
    for z in range(43,54):
        for y in range(45,56):
            for x in range(59,70):
                if density_node(x,y,z)>1.15:
                    wet.append((-2.08+x/127*4.16,-1.12+y/159*5.2,-1.56+z/95*3.12))
    assert wet, ('off-grid droplet disappeared',offset)
    assert max(math.dist((px,py,pz),point) for point in wet)<.04
print('PASS: small isolated spray remains visible at four sub-voxel offsets, radius < 0.04',flush=True)
count=0;run('volume','atlas',{'positions':'pos','lambdas':'lambda'});assert max(read('atlas')[::4])==0
print('PASS: empty volume clears without stale water',flush=True)
print('All native GPU checks passed.',flush=True)
