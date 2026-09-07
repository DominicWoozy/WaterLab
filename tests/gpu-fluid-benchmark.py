"""GPU query timings on the native harness; these are not browser frame rates."""
from pathlib import Path
# Reuse context and production dispatch setup, without running correctness scenarios.
exec(Path(__file__).with_name('gpu-fluid-native.py').read_text().split("run('initialize','pos');initial=")[0])
api('glGenQueries',None,I,c.POINTER(U));api('glBeginQuery',None,U,U);api('glEndQuery',None,U)
api('glGetQueryObjectui64v',None,U,U,c.POINTER(c.c_uint64));api('glDeleteQueries',None,I,c.POINTER(U))
query_log=[]; samples={}; collecting=False

def timed(name,fn):
    if not collecting:return fn()
    q=U();G.glGenQueries(1,c.byref(q));G.glBeginQuery(0x88BF,q);result=fn();G.glEndQuery(0x88BF)
    query_log.append((name,q));return result
raw_run=run

def run(name,out,inputs={},values={}):
    if name in ['key','sort','ranges']:return raw_run(name,out,inputs,values)
    return timed(name,lambda:raw_run(name,out,inputs,values))
raw_grid=grid

def grid(p):return timed('grid',lambda:raw_grid(p))
run('initialize','pos')
for frame in range(80):
    collecting=frame>=20
    step(frame/60)
    run('geometry','geometry',{'positions':'pos','sortedKeys':'keys','cellRanges':'ranges'})
    run('volume','atlas',volume_inputs)
    G.glFinish()
    for name,q in query_log:
        result=c.c_uint64();G.glGetQueryObjectui64v(q,0x8866,c.byref(result));samples.setdefault(name,[]).append(result.value/1e6)
        G.glDeleteQueries(1,c.byref(q))
    query_log.clear()
result={name:round(sum(values)/60,4) for name,values in samples.items()}
result['total']=round(sum(result.values()),4)
print('BENCHMARK',json.dumps({'particles':quality,'gpu_ms_per_step_and_surface':result}),flush=True)
