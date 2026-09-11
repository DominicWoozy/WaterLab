// Shader-only experiment: reuse the terminal segment's opaque intersection.
export function cachedReceiver(shader) {
  const replace = (a, b) => {
    if (!shader.includes(a)) throw Error('Shader changed: ' + a);
    shader = shader.replaceAll(a, b);
  };
  replace(
    'events:u32, complete:u32}',
    'events:u32, complete:u32, receiver:vec4f}',
  );
  replace(
    'var end=0.;var refresh=true;',
    'var receiver=vec4f(0.,-2.,0.,0.);var end=0.;var refresh=true;',
  );
  replace(
    'end=min(end,duckTrace(ro,rd,end).x);',
    'let floorT=(-.97-ro.y)/rd.y;receiver=duckTrace(ro,rd,select(1e5,floorT,floorT>0.));end=min(end,receiver.x);',
  );
  replace(
    'WaterPath(ro,rd,path,weight,events,1u)',
    'WaterPath(ro,rd,path,weight,events,1u,receiver)',
  );
  replace(
    'WaterPath(ro,rd,path,weight,events,0u)',
    'WaterPath(ro,rd,path,weight,events,0u,vec4f(0.,-2.,0.,0.))',
  );
  replace(
    'WaterPath(ro,rd,path,weight,events,select(1u,0u,inside))',
    'WaterPath(ro,rd,path,weight,events,select(1u,0u,inside),vec4f(0.,-2.,0.,0.))',
  );
  replace(
    'var background=scene(path.origin,path.direction);\n  // A bounded unresolved internal path must not leak straight through water.\n  if(path.complete==0u){background=sky(path.direction);}',
    `var background=sky(path.direction);
  if(path.complete!=0u){
   if(path.receiver.y>=0.){background=duckShade(path.receiver,path.direction);}
   else if(path.receiver.y== -1.){background=room(path.origin,path.direction);}
   else{background=scene(path.origin,path.direction);}
  }`,
  );
  return shader;
}
