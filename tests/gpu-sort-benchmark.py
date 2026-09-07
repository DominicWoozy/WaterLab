"""Interleaved original/fused sort timing on exactly the same particle state."""
from pathlib import Path
exec(Path(__file__).with_name('gpu-fluid-benchmark.py').read_text().split('query_log=[]')[0])
run('initialize','pos')
samples={'original':[],'fused':[]}
def original_grid():
    run('key','keys',{'positions':'pos'})
    stage=2
    while stage<=sort_count:
        stride=stage//2
        while stride:
            run('sort','keytmp',{'sortedKeys':'keys'},{'stage':stage,'stride':stride})
            T['keys'],T['keytmp']=T['keytmp'],T['keys'];stride//=2
        stage*=2
    run('ranges','ranges',{'sortedKeys':'keys'})
for frame in range(40):
    for name in (['original','fused'] if frame%2 else ['fused','original']):
        q=U();G.glGenQueries(1,c.byref(q));G.glBeginQuery(0x88BF,q)
        original_grid() if name=='original' else grid('pos')
        G.glEndQuery(0x88BF);G.glFinish()
        result=c.c_uint64();G.glGetQueryObjectui64v(q,0x8866,c.byref(result));G.glDeleteQueries(1,c.byref(q))
        if frame>=10:samples[name].append(result.value/1e6)
original_grid();expected=list(read('keys'));grid('pos');assert list(read('keys'))==expected
print('SORT_BENCHMARK',json.dumps({'particles':quality,'gpu_ms':{k:sum(v)/len(v) for k,v in samples.items()},'identical_keys':True}),flush=True)
