"""GPU regression for planar-wall locking, slip, and solid density support.
--without-wall-support runs the previous boundary behavior for diagnosis.
"""
from pathlib import Path
exec(Path(__file__).with_name('gpu-fluid-native.py').read_text().split("run('initialize','pos');initial=")[0])

def upload(name,data):
    G.glBindTexture(0x0DE1,T[name][0]);G.glTexSubImage2D(0x0DE1,0,0,0,256,128,0x1908,0x1406,data)

baseline='--without-wall-support' in sys.argv
count=1200
for axis,sign in [(0,1),(0,-1),(2,1),(2,-1)]:
    state=(F*(256*128*4))()
    for i in range(count):
        p=[0.,-.2+(i//40)*.028,0.]
        p[axis]=sign*(1.78 if axis==0 else 1.28)
        p[2 if axis==0 else 0]=-.55+(i%40)*1.1/39
        state[i*4:i*4+4]=[*p,1.]
    upload('pos',state);upload('vel',(F*(256*128*4))())
    for frame in range(180):step(frame/60)
    p=read('pos');v=read('vel')
    attached=sum(sign*p[i*4+axis]>(1.78 if axis==0 else 1.28)-.006 for i in range(count))
    elevated=sum(p[i*4+1]>-.3 for i in range(count))
    assert all(math.isfinite(x) for x in p)
    if not baseline:
        assert attached<count*.05,(axis,sign,attached)
        assert elevated<count*.05,(axis,sign,elevated)
    print('WALL_PATCH',json.dumps({'quality':quality,'axis':axis,'sign':sign,'baseline':baseline,'attached':attached,'elevated':elevated,'particles':count}),flush=True)

# An isolated drop on a vertical wall must slide downward, without tangential drag.
count=1;p=(F*(256*128*4))();v=(F*(256*128*4))()
p[:4]=[1.78,1.,0.,1.];v[:4]=[0.,-.4,.6,0.]
upload('pos',p);upload('vel',v)
for frame in range(10):step(frame/60)
p=read('pos');v=read('vel')
assert p[1]<.9 and p[2]>.08 and v[1]<-1.5 and v[2]>.58
assert abs(p[0]-1.78)<1e-5
print('PASS: isolated wall drop retains tangential motion and falls under gravity',flush=True)

# A stationary isolated drop must not be accelerated by an artificial wall repulsion.
upload('pos',(F*(256*128*4))(*[1.78,1.,0.,1.]));upload('vel',(F*(256*128*4))())
for frame in range(10):step(frame/60,gravity=0)
assert max(abs(x) for x in read('vel')[:3])<1e-5
print('PASS: no adhesive force or unconditional wall kick',flush=True)

if not baseline:
    fragment=sources['lambdaFragment'].split('void main(){')[0]+'void main(){result=wallSupport(readAt(positions,id()).xyz);}'
    programs['wallProbe']=compile_program(sources['computeVertex'],fragment)
    # Numerical cap integration independent of the production polynomial.
    for distance in [0.,.15,.45,.8,1.01]:
        p=(F*(256*128*4))();p[:4]=[1.78-distance*.17*scale,.5,0.,1.]
        upload('pos',p);run('wallProbe','lambda',{'positions':'pos'});wall=read('lambda')
        a=min(distance,1);n=4000;dr=(1-a)/n
        cap=15*sum((r*r-a*r)*(1-r)**2*dr for r in (a+(i+.5)*dr for i in range(n)))
        assert abs(wall[3]/3.6-cap)<1e-5,(distance,wall[3]/3.6,cap)
        assert wall[0]<=1e-5 and abs(wall[1])+abs(wall[2])<1e-5
    print('PASS: GPU solid support agrees with independent half-space integration',flush=True)
