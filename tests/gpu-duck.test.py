"""Exercise production GPU fluid/rigid-body coupling without browser automation."""
from pathlib import Path
prefix=Path(__file__).with_name('gpu-fluid-native.py').read_text().split("run('initialize','pos');initial=")[0]
exec(compile(prefix,__file__,'exec'))
for name in ['duckInitialize','duckPredict','duckReduce','duckIntegrate']:
 programs[name]=compile_program(sources['computeVertex'],sources[name+'Fragment'])
T['duck']=target(4,1);T['ducktmp']=target(4,1)
for name in ['linear','angular','linearTmp','angularTmp']:T[name]=target()
for size in [128,64,32,16,8,4,2,1]:
 T['linear'+str(size)]=target(size,max(1,size//2));T['angular'+str(size)]=target(size,max(1,size//2))
original_run=run
iteration=0
def mrt(name,outputs,inputs,values={}):
 G.glBindFramebuffer(0x8D40,T[outputs[0]][1])
 for i,out in enumerate(outputs[1:],1):G.glFramebufferTexture2D(0x8D40,0x8CE0+i,0x0DE1,T[out][0],0)
 G.glDrawBuffers(len(outputs),(U*len(outputs))(*[0x8CE0+i for i in range(len(outputs))]))
 original_run(name,outputs[0],inputs,values)
 for i in range(1,len(outputs)):G.glFramebufferTexture2D(0x8D40,0x8CE0+i,0x0DE1,0,0)
 G.glDrawBuffers(1,(U*1)(0x8CE0))
def run(name,out,inputs={},values={}):
 global iteration
 inputs={**inputs,'duckState':'duck'};values={**values,'duckEnabled':1}
 if name in ['correct','divergenceProject']:
  values['reactionReset']=int(iteration==0);inputs.update(linearSource='linear',angularSource='angular')
  mrt(name,[out,'linearTmp','angularTmp'],inputs,values)
  T['linear'],T['linearTmp']=T['linearTmp'],T['linear'];T['angular'],T['angularTmp']=T['angularTmp'],T['angular'];iteration+=1
 else:original_run(name,out,inputs,values)
original_step=step
def step(t=0,gravity=9.8,splash=(0,0,0)):
 global iteration
 original_run('duckPredict','ducktmp',{'duckState':'duck'},{'dt':1/60,'gravity':gravity});T['duck'],T['ducktmp']=T['ducktmp'],T['duck']
 iteration=0;original_step(t,gravity,splash)
 linear,angular='linear','angular'
 for size in [128,64,32,16,8,4,2,1]:
  nextL,nextA='linear'+str(size),'angular'+str(size)
  mrt('duckReduce',[nextL,nextA],{'linearSource':linear,'angularSource':angular},{'firstLevel':int(size==128)})
  linear,angular=nextL,nextA
 original_run('duckIntegrate','ducktmp',{'duckState':'duck','linearSource':linear,'angularSource':angular});T['duck'],T['ducktmp']=T['ducktmp'],T['duck']

def reset():
 run('initialize','pos');original_run('duckInitialize','duck')
 for name in ['vel','veltmp']:
  G.glBindFramebuffer(0x8D40,T[name][1]);G.glClearColor(0,0,0,0);G.glClear(0x4000)
reset();history=[]
for i in range(600):
 step(i/60);d=list(read('duck'));history.append(d)
 assert all(math.isfinite(v) for v in d), (i,d)
 if i%60==59:print('step',i+1,'position',d[:3],'q',d[4:8],'v',d[8:11],'w',d[12:15],flush=True)
 if max(abs(x) for x in d)>100:raise AssertionError(('unstable',i,d))
steady=history[-120:];height=sum(d[1] for d in steady)/len(steady);speed=sum(sum(v*v for v in d[8:11]) for d in steady)/len(steady)
print('steady height',height,'speed^2',speed,flush=True)
assert height>-.76,('duck sank',height)
assert speed<.2,('rest jitter',speed)
assert max(d[1] for d in steady)-min(d[1] for d in steady)<.12
assert all(abs(sum(x*x for x in d[4:8])-1)<1e-4 for d in history)
print('PASS: buoyancy emerges from pressure, calm settling, unit orientation',flush=True)
rest=history[-1]
for i in range(90):step(10+i/60,splash=(rest[0]-.2,rest[2],2.) if i==0 else (0,0,0))
wave=list(read('duck'));assert math.dist(rest[:3],wave[:3])>.01
print('PASS: nearby splash moves rigid body',wave[:3],flush=True)
count=0
for i in range(300):step(12+i/60)
d=list(read('duck'));print('drained position',d[:3],flush=True)
assert -.99<d[1]<-.65 and abs(d[9])<.15,('empty tank contact',d)
print('PASS: draining removes buoyancy; duck rests on tank floor',flush=True)
# An isolated slip contact exchanges equal/opposite linear and angular impulse.
# Gravity, pressure, global damping and tank contacts are absent in this check.
count=1
for name in ['pos','vel','lambda','linear','angular']:
 G.glBindFramebuffer(0x8D40,T[name][1]);G.glClearColor(0,0,0,0);G.glClear(0x4000)
def write_first(name,values):
 G.glBindTexture(0x0DE1,T[name][0]);a=(F*len(values))(*values);G.glTexSubImage2D(0x0DE1,0,0,0,len(values)//4,1,0x1908,0x1406,a)
write_first('duck',[0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0])
point=[.37,.055,0];before=[-.4,.2,0]
write_first('pos',point+[1]);write_first('vel',before+[0]);grid('pos')
run('divergenceProject','veltmp',{'positions':'pos','velocities':'vel','lambdas':'lambda','sortedKeys':'keys','cellRanges':'ranges'})
after=list(read('veltmp'))[:3];linear=list(read('linear'))[:3];angular=list(read('angular'))[:3]
pm=(2*math.pi/15)*(.17*scale)**3/3.6
delta=[pm*(after[i]-before[i]) for i in range(3)]
assert math.sqrt(sum((delta[i]+linear[i])**2 for i in range(3)))<1e-8
moment=[point[1]*delta[2]-point[2]*delta[1],point[2]*delta[0]-point[0]*delta[2],point[0]*delta[1]-point[1]*delta[0]]
assert math.sqrt(sum((moment[i]+angular[i])**2 for i in range(3)))<1e-8
assert after[0]>-.01 and 0<after[1]<before[1]
print('PASS: isolated moving-boundary contact conserves linear/angular impulse',flush=True)
