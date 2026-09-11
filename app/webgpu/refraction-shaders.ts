// Bounded camera transmission through the density isosurface and a primary
// analytic drop. Opaque receivers terminate a segment before any later exit.
export const transmissionTrace = /* wgsl */ `
struct WaterPath {origin:vec3f, direction:vec3f, distance:f32, weight:f32, events:u32, complete:u32}
fn interfaceReflectance(cosine:f32,eta:f32)->f32 {
 let ci=clamp(cosine,0.,1.);let sin2=eta*eta*(1.-ci*ci);
 if(sin2>=1.){return 1.;}let ct=sqrt(1.-sin2);
 let rs=(eta*ci-ct)/max(eta*ci+ct,1e-8);
 let rp=(ci-eta*ct)/max(ci+eta*ct,1e-8);
 return .5*(rs*rs+rp*rp);
}
// These planes truncate the reconstructed field at the tank. They are not
// free water/air interfaces. The sides are decorative transparent boundaries,
// not a glass solid; the tile receiver is opaque. Keep the tolerance close to one
// refinement interval, so a detached surface near a wall still refracts.
fn tankContact(p:vec3f,rd:vec3f)->u32 {
 let epsilon=.0003;
 if(rd.y<0. && abs(p.y+.96)<epsilon){return 1u;}
 if(p.y>=-.96-epsilon && p.y<.8){
  if(p.x*rd.x>0. && abs(abs(p.x)-1.86)<epsilon){return 2u;}
  if(p.z*rd.z>0. && abs(abs(p.z)-1.36)<epsilon){return 2u;}
 }
 return 0u;
}
fn traceTransmission(start:vec3f,direction:vec3f,startInside:bool,detailId:u32)->WaterPath {
 var ro=start;var rd=direction;var inside=startInside;var analytic=detailId;
 var path=0.;var weight=1.;var events=0u;var at=0.;
 var end=0.;var refresh=true;
 // Fixed spatial steps retain resolved thin features; a global budget bounds
 // worst-case work even for repeated crossings or total internal reflection.
 for(var sample=0;sample<256;sample++){
  if(refresh){
   end=max(0.,boxHit(ro,rd).y);
   end=min(end,lightReceiver(ro,rd).x);
   end=min(end,duckTrace(ro,rd,end).x);
   at=0.;refresh=false;
  }
  if(analytic>0u){
   let d=details[analytic-1u];let roots=detailRoots(d,ro,rd);let length=max(0.,roots.y);
   if(end<length){path+=end;return WaterPath(ro,rd,path,weight,events,1u);}
   path+=length;let hit=ro+rd*length;
   let face=-normalize(detailMetric(d)*(hit-d.center.xyz));
   let outgoing=refract(rd,face,1.333);events++;
   if(dot(outgoing,outgoing)<1e-8){
    rd=normalize(reflect(rd,face));ro=hit+face*.00002;
   }else{
    weight*=1.-interfaceReflectance(dot(-rd,face),1.333);
    rd=normalize(outgoing);ro=hit-face*.00002;inside=false;analytic=0u;
   }
   if(events>=4u){return WaterPath(ro,rd,path,weight,events,select(1u,0u,inside));}
   refresh=true;continue;
  }
  if(at>=end){return WaterPath(ro,rd,path,weight,events,1u);}
  let stepSize=select(.026,.018,inside);
  let next=min(end,at+stepSize);
  let value=density(ro+rd*next);let nextInside=value>S.config.x;
  if(nextInside!=inside){
   var lo=at;var hi=next;
   for(var refine=0;refine<6;refine++){
    let mid=(lo+hi)*.5;
    if((density(ro+rd*mid)>S.config.x)==inside){lo=mid;}else{hi=mid;}
   }
   let hitDistance=(lo+hi)*.5;let hit=ro+rd*hitDistance;
   if(inside){path+=hitDistance-at;}
   if(inside){
    let contact=tankContact(hit,rd);
    if(contact>0u){
     // Close only the artificial bottom gap, stopping at the already tested
     // nearest opaque receiver. Side transmission keeps the incoming segment.
     if(contact==1u){path+=max(0.,end-hitDistance);}
     return WaterPath(ro,rd,path,weight,events,1u);
    }
   }
   var face=normalAt(hit);if(dot(face,rd)>0.){face=-face;}
   let eta=select(1./1.333,1.333,inside);let outgoing=refract(rd,face,eta);events++;
   if(dot(outgoing,outgoing)<1e-8){rd=normalize(reflect(rd,face));ro=hit+face*.0005;}
   else{
    weight*=1.-interfaceReflectance(dot(-rd,face),eta);
    rd=normalize(outgoing);ro=hit-face*.0005;inside=!inside;
   }
   if(events>=4u){return WaterPath(ro,rd,path,weight,events,select(1u,0u,inside));}
   refresh=true;
  }else{
   if(inside){path+=next-at;}
   at=next;
  }
 }
 return WaterPath(ro,rd,path,weight,events,0u);
}
`;
