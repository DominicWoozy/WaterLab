"""Isolate GPU draw cost from Python uniform submission; diagnostic replay only."""
from pathlib import Path
exec(compile(Path(__file__).with_name('gpu-duck.test.py').read_text().split('reset();history=[]')[0],__file__,'exec'))
import statistics
api('glGenQueries',None,I,c.POINTER(U));api('glBeginQuery',None,U,U);api('glEndQuery',None,U)
api('glGetQueryObjectui64v',None,U,U,c.POINTER(c.c_uint64));api('glDeleteQueries',None,I,c.POINTER(U))
raw_run=original_run;collecting=False;samples={}
def profiled_run(name,out,inputs={},values={}):
    raw_run(name,out,inputs,values)
    if not collecting:return
    q=U();G.glGenQueries(1,c.byref(q));G.glBeginQuery(0x88BF,q)
    for _ in range(16):
        if name=='volume':
            G.glClear(0x4000);G.glEnable(0x0BE2);G.glBlendFunc(1,1);G.glDrawArraysInstanced(4,0,6,count*20);G.glDisable(0x0BE2)
        elif name=='radixScatter':G.glDrawArrays(0,0,sort_count)
        else:G.glDrawArrays(4,0,3)
    G.glEndQuery(0x88BF);G.glFinish()
    result=c.c_uint64();G.glGetQueryObjectui64v(q,0x8866,c.byref(result));G.glDeleteQueries(1,c.byref(q))
    samples.setdefault(name,[]).append(result.value/1e6/16)
original_run=profiled_run
reset()
for frame in range(48):
    collecting=frame>=24
    step(frame/60)
    run('geometry','geometry',{'positions':'pos','sortedKeys':'keys','cellRanges':'ranges'})
    run('volume','atlas',volume_inputs);filter_surface()
# Some stages occur repeatedly per step; account for every iteration.
result={k:round(sum(v)/24,4) for k,v in samples.items()}
print('BATCHED_STAGE_GPU_MS',json.dumps({'particles':quality,'gpu_draw_ms_per_step':result,'sum':round(sum(result.values()),4)}),flush=True)
