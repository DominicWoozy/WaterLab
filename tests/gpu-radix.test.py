"""Production radix shaders vs CPU stable order and prior GPU sort, including padding transitions."""
from pathlib import Path
import random
from bisect import bisect_left
exec(compile(Path(__file__).with_name('gpu-fluid-native.py').read_text().split("run('initialize','pos');initial=")[0], __file__, 'exec'))
rng=random.Random(817)

def upload_positions(points):
    data=(F*(32768*4))()
    for i,p in enumerate(points):
        for j in range(3):data[4*i+j]=p[j]
        data[4*i+3]=1
    G.glBindTexture(0x0DE1,T['pos'][0])
    G.glTexSubImage2D(0x0DE1,0,0,0,256,128,0x1908,0x1406,data)

fixtures=[(n,'random') for n in [0,1,31,32,33,15000,16384,16385,30000]]
fixtures += [(30000,'one_cell'),(30000,'walls'),(15000,'random'),(0,'random')]
for count,kind in fixtures:
    points=[]
    for i in range(count):
        if kind=='one_cell':p=(0.,0.,0.)
        elif kind=='walls':p=((-1.78,1.78)[i&1],(-.917,3.8)[(i>>1)&1],(-1.28,1.28)[(i>>2)&1])
        else:p=(rng.uniform(-1.78,1.78),rng.uniform(-.917,3.8),rng.uniform(-1.28,1.28))
        points.append(p)
    upload_positions(points)
    sort_count=16384 if count<=16384 else 32768
    run('key','keys',{'positions':'pos'})
    raw=read('keys');expected=sorted(tuple(raw[i*4:i*4+2]) for i in range(sort_count))
    grid('pos');actual=read('keys')
    assert [tuple(actual[i*4:i*4+2]) for i in range(sort_count)]==expected,(count,kind)
    assert all(actual[i*4+2]==actual[i*4+3]==0 for i in range(sort_count))
    ranges=list(read('ranges'));key_values=[p[0] for p in expected]
    for k in range(30720):
        assert ranges[4*k:4*k+2]==[bisect_left(key_values,k),bisect_left(key_values,k+1)],(count,kind,k)
    bitonic_grid('pos')
    assert list(read('keys'))[:sort_count*4]==list(actual)[:sort_count*4],(count,kind)
    assert list(read('ranges'))==ranges,(count,kind)
    print('PASS:',count,kind,'stable keys, complete permutation, every cell range; identical to bitonic',flush=True)
