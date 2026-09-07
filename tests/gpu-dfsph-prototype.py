"""Evaluation-only full two-solver DFSPH prototype at the app's fixed timestep.
No warm start, CFL adaptation or boundary-density model. Not shipped as app solver.
Uses the same normalized kernel and equal masses to isolate integration differences.
"""
from pathlib import Path
exec(Path(__file__).with_name('gpu-fluid-evaluation.py').read_text().split('results={}')[0])
programs['externalVelocity']=compile_program(sources['computeVertex'],sources['predictFragment'].replace('result=vec4(bound(p+limited(v,12.)*dt),1.);','result=vec4(limited(v,12.),0.);'))
programs['densityResidual']=compile_program(sources['computeVertex'],sources['divergenceResidualFragment'].replace('uniform sampler2D factors;','uniform sampler2D factors;uniform float dt;').replace('max(rate,0.)*f.x','max((f.y-1.)/dt+rate,0.)*f.x'))
programs['advect']=compile_program(sources['computeVertex'],sources['velocityFragment'].replace('vec3 v=(readAt(positions,i).xyz-readAt(oldPositions,i).xyz)/dt;','vec3 v=readAt(velocities,i).xyz;').replace('if(readAt(oldPositions,i).w<.5)v=vec3(0.,-1.4,0.);','').replace('result=vec4(limited(v,12.)*.998,0.);','result=vec4(bound(readAt(positions,i).xyz+v*dt),1.);'))
run('initialize','pos');grid('pos');reorder(False,count)
ni={'positions':'pos','cellRanges':'ranges'}
run('divergenceFactor','factor',ni)
for frame in range(180):
    run('externalVelocity','veltmp',{'positions':'pos','velocities':'vel'},{'dt':1/60,'time':frame/60,'gravity':9.8,'previousCount':count})
    run('viscosity','vel',{**ni,'velocities':'veltmp'},{'viscosity':.025})
    for _ in range(4):
        run('densityResidual','lambda',{**ni,'velocities':'vel','factors':'factor'},{'dt':1/60})
        run('divergenceProject','veltmp',{**ni,'velocities':'vel','lambdas':'lambda'})
        T['vel'],T['veltmp']=T['veltmp'],T['vel']
    run('advect','pred',{'positions':'pos','velocities':'vel'},{'dt':1/60})
    T['pos'],T['pred']=T['pred'],T['pos']
    grid('pos');reorder(False,count)
    run('divergenceFactor','factor',ni)
    for _ in range(2):
        run('divergenceResidual','lambda',{**ni,'velocities':'vel','factors':'factor'})
        run('divergenceProject','veltmp',{**ni,'velocities':'vel','lambdas':'lambda'})
        T['vel'],T['veltmp']=T['veltmp'],T['vel']
    if frame in [59,119,179]:
        print('DFSPH_PROTOTYPE',json.dumps({'particles':quality,'time':(frame+1)/60,**metrics()}),flush=True)
print('Evaluation finished; this prototype does not establish convergence or production readiness.',flush=True)
