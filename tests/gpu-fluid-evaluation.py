"""Controlled PBF / PBF+DFSPH projection and raw / filtered surface comparisons.
Uses production GLSL on the native GPU; no browser or visual QA.
"""
from pathlib import Path
exec(Path(__file__).with_name('gpu-fluid-native.py').read_text().split("run('initialize','pos');initial=")[0])

def upload(name,data):
    G.glBindTexture(0x0DE1,T[name][0]);G.glTexSubImage2D(0x0DE1,0,0,0,256,128,0x1908,0x1406,data)

def metrics():
    grid('pos');reorder(False,count)
    ni={'positions':'pos','cellRanges':'ranges'}
    run('divergenceFactor','factor',ni)
    run('divergenceResidual','lambda',{**ni,'velocities':'vel','factors':'factor'})
    f=read('factor');r=read('lambda');v=read('vel')
    compression=[max(0,f[i*4+1]-1) for i in range(count)]
    return {'compression_mean_pct':100*sum(compression)/count,
            'compression_p95_pct':100*sorted(compression)[int(count*.95)],
            'compression_max_pct':100*max(compression),
            'positive_density_rate':sum(max(0,r[i*4+1]) for i in range(count))/count,
            'mean_squared_speed':sum(v[i*4+j]**2 for i in range(count) for j in range(3))/count}

def surface(atlas):
    def node(x,y,z):return atlas[4*((z//8*160+y)*1024+z%8*128+x)]
    def field(x,y,z):
        u=(x+2.08)/4.16*127;v=(y+1.12)/5.2*159;w=(z+1.56)/3.12*95
        ix,iy,iz=math.floor(u),math.floor(v),math.floor(w);fx,fy,fz=u-ix,v-iy,w-iz
        return sum(node(ix+dx,iy+dy,iz+dz)*(fx if dx else 1-fx)*(fy if dy else 1-fy)*(fz if dz else 1-fz) for dx in range(2) for dy in range(2) for dz in range(2))
    heights=[]
    for iz in range(25):
        for ix in range(33):
            x=-.8+ix*.05;z=-.6+iz*.05;hi=-.15;lo=hi
            for _ in range(100):
                lo-=.008
                if field(x,lo,z)>1.15:break
                hi=lo
            assert hi>-.9,'hole in connected water surface'
            for _ in range(9):
                mid=(lo+hi)/2
                if field(x,mid,z)>1.15:lo=mid
                else:hi=mid
            heights.append((lo+hi)/2)
    mean=sum(heights)/len(heights)
    curvature=[heights[z*33+x-1]+heights[z*33+x+1]+heights[(z-1)*33+x]+heights[(z+1)*33+x]-4*heights[z*33+x] for z in range(1,24) for x in range(1,32)]
    return {'height_rms_mm':1000*math.sqrt(sum((h-mean)**2 for h in heights)/len(heights)),
            'corrugation_mm':1000*math.sqrt(sum(h*h for h in curvature)/len(curvature)),
            'mean_height':mean}, heights

results={}
for projection in [False,True]:
    run('initialize','pos');upload('vel',(F*(256*128*4))())
    history=[]
    for frame in range(180):
        step(frame/60,projection=projection)
        if frame in [59,119,179]:history.append(metrics())
    run('geometry','geometry',{'positions':'pos','cellRanges':'ranges'})
    run('volume','atlas',volume_inputs)
    raw,_=surface(read('atlas'));filter_surface();filtered,_=surface(read('surfaceTemp'))
    result={'samples_1_2_3_seconds':history,'raw_surface':raw,'filtered_surface':filtered}
    results['hybrid' if projection else 'pbf']=result
    print('EVALUATION',json.dumps({'particles':quality,'projection':projection,**result}),flush=True)
assert results['hybrid']['samples_1_2_3_seconds'][-1]['mean_squared_speed']<results['pbf']['samples_1_2_3_seconds'][-1]['mean_squared_speed']*.5
assert results['hybrid']['samples_1_2_3_seconds'][-1]['compression_mean_pct']<results['pbf']['samples_1_2_3_seconds'][-1]['compression_mean_pct']
assert results['hybrid']['filtered_surface']['corrugation_mm']<results['hybrid']['raw_surface']['corrugation_mm']*.85

# A known broad wave must survive the filter. It must not flatten all dynamics.
run('initialize','pos');p=read('pos')
for i in range(count):p[4*i+1]+=.065*math.sin(p[4*i]*3)*math.cos(p[4*i+2]*2)
upload('pos',p);grid('pos');reorder(False,count)
run('geometry','geometry',{'positions':'pos','cellRanges':'ranges'});run('volume','atlas',volume_inputs)
raw,h0=surface(read('atlas'));filter_surface();filtered,h1=surface(read('surfaceTemp'))
amplitude=(max(h1)-min(h1))/(max(h0)-min(h0))
assert .90<amplitude<1.10,amplitude
assert abs(filtered['mean_height']-raw['mean_height'])<.012
print('PASS: broad-wave amplitude retained %.2f%%, mean surface shift %.2f mm'%(amplitude*100,(filtered['mean_height']-raw['mean_height'])*1000),flush=True)
