// Shade the opaque background only when the primary water ray did not hit.
export function deferredWaterShading(shader) {
  const replace = (a, b) => {
    if (!shader.includes(a)) throw Error('Shader changed: ' + a);
    shader = shader.replace(a, b);
  };
  replace(
    'var color=room(S.eye.xyz,rd);if(primaryDuck.y>=0.){color=duckShade(primaryDuck,rd);}',
    'var color=vec3f(0.);',
  );
  replace(
    'color=waterColor(p,n,rd,0u);\n }',
    'color=waterColor(p,n,rd,0u);\n }else{if(primaryDuck.y>=0.){color=duckShade(primaryDuck,rd);}else{color=room(S.eye.xyz,rd);}}',
  );
  replace(
    'var background=scene(path.origin,path.direction);\n  // A bounded unresolved internal path must not leak straight through water.\n  if(path.complete==0u){background=sky(path.direction);}',
    'var background=vec3f(0.);\n  if(path.complete==0u){background=sky(path.direction);}else{background=scene(path.origin,path.direction);}',
  );
  return shader;
}
