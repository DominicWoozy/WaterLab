"""Interleaved production bitonic/radix grid timings; native GPU, not browser FPS."""
from pathlib import Path
exec(Path(__file__).with_name('gpu-fluid-benchmark.py').read_text().split('query_log=[]')[0])
run('initialize','pos')
samples={'bitonic_fused':[],'radix':[]}
for frame in range(48):
    for name in (['bitonic_fused','radix'] if frame%2 else ['radix','bitonic_fused']):
        q=U();G.glGenQueries(1,c.byref(q));G.glBeginQuery(0x88BF,q)
        bitonic_grid('pos') if name=='bitonic_fused' else grid('pos')
        G.glEndQuery(0x88BF);G.glFinish()
        result=c.c_uint64();G.glGetQueryObjectui64v(q,0x8866,c.byref(result));G.glDeleteQueries(1,c.byref(q))
        if frame>=12:samples[name].append(result.value/1e6)
bitonic_grid('pos');expected=list(read('keys'))[:sort_count*4];expected_ranges=list(read('ranges'))
grid('pos');assert list(read('keys'))[:sort_count*4]==expected
assert list(read('ranges'))==expected_ranges
print('SORT_BENCHMARK',json.dumps({'particles':quality,'gpu_ms':{k:sum(v)/len(v) for k,v in samples.items()},'identical_keys_and_ranges':True}),flush=True)
