// Local experiments only, deliberately not imported by the application.
// Inspired by pixel-footprint filtering in Kaplanyan et al. (HPG 2016).
// This is quadrature of the current Phong lobe, NOT their GGX/Beckmann NDF method.
import assert from 'node:assert/strict';
import { renderShader } from '../../app/webgpu/render-shaders.ts';
export function appearanceShader(
  filtered,
  mode = 'baseline',
  highlights = false,
) {
  let code = renderShader(filtered);
  const change = (a, b) => {
    assert.ok(code.includes(a), `missing shader fragment: ${a}`);
    code = code.replace(a, b);
  };
  if (mode === 'aa') {
    change(
      'struct FragmentOut {',
      `var<private> reflectionDx:vec3f;
var<private> reflectionDy:vec3f;
fn pixelSun(r:vec3f,sun:vec3f,detailId:u32)->f32 {
 if(detailId>0u){return pow(max(dot(r,sun),0.),240.);}
 let x=reflectionDx*.2886751346;let y=reflectionDy*.2886751346;
 return .25*(pow(max(dot(normalize(r+x+y),sun),0.),240.)+
 pow(max(dot(normalize(r+x-y),sun),0.),240.)+
 pow(max(dot(normalize(r-x+y),sun),0.),240.)+
 pow(max(dot(normalize(r-x-y),sun),0.),240.));
}
struct FragmentOut {`,
    );
    change(
      ' if(found){\n  for(var i=0;i<7;i++)',
      ' var p=S.eye.xyz+rd*at;var n=vec3f(0.,1.,0.);\n if(found){\n  for(var i=0;i<7;i++)',
    );
    change(
      '  let p=S.eye.xyz+rd*at;var n=normalAt(p);if(dot(n,rd)>0.){n=-n;}\n  color=waterColor',
      `  p=S.eye.xyz+rd*at;n=normalAt(p);if(dot(n,rd)>0.){n=-n;}
 }
 // Derivatives execute in uniform control flow, including dry pixels.
 let reflected=reflect(rd,n);reflectionDx=dpdx(reflected);reflectionDy=dpdy(reflected);
 if(found){
  color=waterColor`,
    );
    change(
      'pow(max(dot(reflect(rd,n),sun),0.),240.)',
      'pixelSun(reflect(rd,n),sun,detailId)',
    );
  }
  if (mode === 'srgb') {
    change(
      'return FragmentOut(vec4f(pow(color,vec3f(.91)),1.),depth);',
      'return FragmentOut(vec4f(select(color*12.92,1.055*pow(max(color,vec3f(0.)),vec3f(1./2.4))-.055,color>vec3f(.0031308)),1.),depth);',
    );
    change(
      'pow(textureSampleLevel(albedo,albedoSampler,uv,0.).rgb,vec3f(2.2))',
      'srgbToLinear(textureSampleLevel(albedo,albedoSampler,uv,0.).rgb)',
    );
    change(
      'fn duckShade(',
      'fn srgbToLinear(c:vec3f)->vec3f{return select(c/12.92,pow((c+.055)/1.055,vec3f(2.4)),c>vec3f(.04045));}\nfn duckShade(',
    );
  }
  if (highlights) {
    const start = code.indexOf('  let refracted=refract(rd,n,1./1.333);');
    const end = code.indexOf('\n return color;\n}', start);
    assert.ok(start >= 0 && end > start);
    code =
      code.slice(0, start) +
      ` let sun=normalize(vec3f(-.6,1.,.35));\n var color=vec3f(${mode === 'aa' ? 'pixelSun(reflect(rd,n),sun,detailId)' : 'pow(max(dot(reflect(rd,n),sun),0.),240.)'});` +
      code.slice(end);
    change(
      '}else{if(primaryDuck.y>=0.){color=duckShade(primaryDuck,rd);}else{color=room(S.eye.xyz,rd);}}',
      '}',
    );
    change(
      ' color=1.-exp(-color*1.35);let vignette=in.uv-.5;color*=1.-.12*dot(vignette,vignette);',
      '',
    );
    change('pow(color,vec3f(.91))', 'color');
  }
  return code;
}
