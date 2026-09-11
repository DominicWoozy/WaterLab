// Optional, bounded secondary rays through the existing reconstructed field.
// No screen-space history and no recursive reflection. Tiny analytic droplets
// are shaded as receivers, but secondary rays only intersect the density field.
export const selfOpticsCommon = /* wgsl */ `
fn reflectedWaterHit(ro:vec3f,rd:vec3f,end:f32)->f32 {
 let interval=boxHit(ro,rd);var at=max(0.,interval.x);let stop=min(end,interval.y);
 if(stop<=at){return -1.;}
 var last=at;var outside=false;
 for(var i=0;i<384;i++){
  let wet=density(ro+rd*at)>S.config.x;
  if(wet&&outside){
   var lo=last;var hi=at;
   for(var j=0;j<6;j++){let mid=(lo+hi)*.5;if(density(ro+rd*mid)>S.config.x){hi=mid;}else{lo=mid;}}
   let hit=(lo+hi)*.5;
   // A clipped solid contact is not a free reflecting water/air interface.
   if(tankContact(ro+rd*hit,-rd)>0u){return -1.;}
   return hit;
  }
  outside=outside||!wet;
  if(at>=stop){break;}last=at;at=min(stop,at+.018);
 }
 return -1.;
}
fn waterSunVisibility(p:vec3f,n:vec3f)->vec3f {
 if(S.right.w<.5||S.light.x<=0.){return vec3f(1.);}
 let sun=normalize(vec3f(-.6,1.,.35));let ro=p+n*.006;
 // Opaque geometry also occludes the direct source; ambient stays available.
 if(duckTrace(ro,sun,1e5).y>=0.){return vec3f(0.);}
 let interval=boxHit(ro,sun);var at=max(0.,interval.x);let stop=interval.y;
 var distance=0.;var energy=1.;var inside=density(ro+sun*at)>S.config.x;
 for(var i=0;i<384;i++){
  if(at>=stop){break;}
  let next=min(stop,at+.022);let wet=density(ro+sun*next)>S.config.x;
  if(wet!=inside){
   var lo=at;var hi=next;
   for(var j=0;j<6;j++){let mid=(lo+hi)*.5;if((density(ro+sun*mid)>S.config.x)==inside){lo=mid;}else{hi=mid;}}
   let hit=(lo+hi)*.5;let point=ro+sun*hit;
   distance+=select(next-hit,hit-at,inside);
   // Straight shadow-ray approximation: extinction and Fresnel transmission,
   // without solving a refracted connection to the sun or focusing caustics.
   if(tankContact(point,select(-sun,sun,inside))==0u){
    let cosine=abs(dot(normalAt(point),sun));
    energy*=1.-(.0204+.9796*pow(1.-cosine,5.));
   }
  }else{if(inside){distance+=next-at;}}
  inside=wet;at=next;
 }
 return exp(-vec3f(1.25,.2,.065)*distance)*energy;
}
fn litWaterReflection(rd:vec3f,visibility:vec3f,hit:vec4f)->vec3f {
 if(hit.y>=0.){return duckShade(hit,rd);}
 let sun=normalize(vec3f(-.6,1.,.35));
 let direct=vec3f(1.,.91,.72)*pow(max(dot(rd,sun),0.),260.)*2.5*S.light.x;
 return max(vec3f(0.),sky(rd)-direct*(1.-visibility));
}
`;

export const selfOpticsWater = /* wgsl */ `
fn waterColor(p:vec3f,n:vec3f,rd:vec3f,detailId:u32)->vec3f {
 let visibility=waterSunVisibility(p,n);var reflection=vec3f(0.);
 if(S.light.y>.5){
  let reflected=reflect(rd,n);let ro=p+n*.006;
  let opaque=duckTrace(ro,reflected,lightReceiver(ro,reflected).x);
  reflection=litWaterReflection(reflected,visibility,opaque);
  if(S.forward.w>.5){
   let at=reflectedWaterHit(ro,reflected,opaque.x);
   if(at>=0.){
    let point=ro+reflected*at;var normal=normalAt(point);
    if(dot(normal,reflected)>0.){normal=-normal;}
    let light=waterSunVisibility(point,normal);let secondary=reflect(reflected,normal);
    let duckHit=duckTrace(point+secondary*.006,secondary,1e5);
    reflection=baseWaterColor(point,normal,reflected,0u,litWaterReflection(secondary,light,duckHit),light);
   }
  }
 }
 return baseWaterColor(p,n,rd,detailId,reflection,visibility);
}
`;
