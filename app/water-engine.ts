export type WaterSettings = { amplitude: number; speed: number; depth: number; light: number; reflection: boolean; caustics: boolean; paused: boolean };
export const defaults: WaterSettings = { amplitude: 0.55, speed: 0.8, depth: 1.05, light: 1.2, reflection: true, caustics: true, paused: false };
const vertex = `attribute vec2 position; void main(){ gl_Position=vec4(position,0.,1.); }`;
const fragment = `
precision highp float;
uniform vec2 resolution;
uniform float time, amplitude, depth, lightPower, reflectionOn, causticsOn;
uniform vec3 eye, target;
uniform vec4 drops[8];
float heightAt(vec2 p){
 float t=time;
 float h=sin(p.x*3.1+p.y*1.8-t*1.5)*.060;
 h+=sin(p.x*-2.4+p.y*4.2-t*1.9)*.036;
 h+=sin(p.x*6.3+p.y*2.7-t*2.5)*.021;
 h+=sin(p.x*8.2-p.y*7.1-t*3.1)*.012;
 h+=sin(p.x*13.1+p.y*9.3-t*3.8)*.005;
 h*=amplitude;
 for(int i=0;i<8;i++){
  float age=time-drops[i].z;
  if(age>=0. && age<9. && drops[i].w>0.){
   float r=length(p-drops[i].xy);
   float front=r-age*.85;
   h+=sin(front*19.)*exp(-front*front*3.5)*exp(-age*.55)*.075*drops[i].w;
  }
 }
 return h;
}
vec2 boxHit(vec3 ro,vec3 rd,vec3 lo,vec3 hi){
 vec3 safe=sign(rd)*max(abs(rd),vec3(.00001));
 vec3 a=(lo-ro)/safe,b=(hi-ro)/safe;
 vec3 mn=min(a,b),mx=max(a,b);
 return vec2(max(max(mn.x,mn.y),mn.z),min(min(mx.x,mx.y),mx.z));
}
vec3 sky(vec3 r){
 float up=smoothstep(-.15,.8,r.y);
 vec3 c=mix(vec3(.055,.12,.15),vec3(.55,.72,.75),up);
 float cloud=sin(r.x*8.+r.z*6.)*.5+sin(r.z*15.-r.x*4.)*.25;
 c+=smoothstep(.16,.7,cloud)*.25*up;
 vec3 sun=normalize(vec3(-.6,1.,.35));
 c+=vec3(1.,.92,.74)*pow(max(dot(r,sun),0.),350.)*3.*lightPower;
 float softbox=pow(max(dot(r,normalize(vec3(.3,.9,-.6))),0.),22.);
 c+=vec3(.65,.86,.9)*softbox*.5*lightPower;
 return c;
}
vec3 tiles(vec2 p){
 vec2 grid=abs(fract(p*3.)-.5);
 float seam=smoothstep(.468,.493,max(grid.x,grid.y));
 float check=mod(floor(p.x*3.)+floor(p.y*3.),2.);
 return mix(mix(vec3(.43,.56,.56),vec3(.53,.64,.62),check),vec3(.16,.28,.29),seam*.58);
}
float caustic(vec2 p){
 vec2 q=p*7.;
 q+=vec2(sin(p.y*4.+time*.9),cos(p.x*3.-time*.7))*.9*amplitude;
 float a=sin(q.x+sin(q.y+time*.55));
 float b=sin(q.y+sin(q.x-time*.45));
 return pow(max(0.,1.-abs(a+b)*.66),13.)*.65;
}
void main(){
 vec2 uv=(gl_FragCoord.xy-.5*resolution)/resolution.y;
 if(resolution.x/resolution.y>1.18) uv.x+=.19;
 vec3 forward=normalize(target-eye),right=normalize(cross(forward,vec3(0.,1.,0.))),up=cross(right,forward);
 vec3 rd=normalize(forward*1.65+uv.x*right+uv.y*up),ro=eye;
 vec3 color=vec3(.034,.052,.060)+vec3(.019,.025,.028)*(1.-length(uv)*.3);
 float floorY=-depth-.13;
 float ground=(floorY-ro.y)/rd.y;
 if(ground>0.){
  vec3 g=ro+rd*ground;
  float d=length(g.xz*vec2(.75,1.));
  color+=vec3(.021,.028,.031)*exp(-d*.22);
  vec2 gg=abs(fract(g.xz*.5+.5)-.5);
  float lines=smoothstep(.495,.5,max(gg.x,gg.y));
  color+=lines*.012*exp(-d*.16);
  float shadow=exp(-max(abs(g.x)-1.7,0.)*2.7-max(abs(g.z)-1.1,0.)*2.7);
  color*=1.-shadow*.55;
 }
 // A solid plinth makes the depth and refracted floor legible.
 vec2 base=boxHit(ro,rd,vec3(-1.9,-depth-.12,-1.4),vec3(1.9,-depth,1.4));
 if(base.x>0. && base.x<base.y){
  vec3 p=ro+rd*base.x;
  color=vec3(.11,.16,.17);
  if(p.y>-depth-.006) color=tiles(p.xz)*.53;
  float edge=min(abs(abs(p.x)-1.9),abs(abs(p.z)-1.4));
  color+=vec3(.12,.19,.19)*exp(-edge*95.);
 }
 vec2 hit=boxHit(ro,rd,vec3(-1.85,-depth,-1.35),vec3(1.85,.5,1.35));
 float dist=max(hit.x,0.); bool found=false;
 if(hit.y>dist){
  for(int i=0;i<72;i++){
   vec3 p=ro+rd*dist;
   float gap=p.y-heightAt(p.xz);
   if(gap<.0015){found=true;break;}
   dist+=max(gap*.43,.004);
   if(dist>hit.y)break;
  }
 }
 if(found){
  vec3 p=ro+rd*dist;
  vec3 n;
  bool side=false;
  if(abs(p.x)>1.848){n=vec3(sign(p.x),0.,0.);side=true;}
  else if(abs(p.z)>1.348){n=vec3(0.,0.,sign(p.z));side=true;}
  else{
   float e=.008;
   n=normalize(vec3(heightAt(p.xz-vec2(e,0.))-heightAt(p.xz+vec2(e,0.)),2.*e,heightAt(p.xz-vec2(0.,e))-heightAt(p.xz+vec2(0.,e))));
  }
  vec3 refr=refract(rd,n,1./1.333);
  vec2 exitHit=boxHit(p+refr*.005,refr,vec3(-1.85,-depth,-1.35),vec3(1.85,.5,1.35));
  float travel=max(exitHit.y,.01);
  vec3 q=p+refr*travel;
  vec3 transmitted;
  if(q.y < -depth+.03){
   transmitted=tiles(q.xz)*(.55+lightPower*.35);
   transmitted+=vec3(.54,.95,.83)*caustic(q.xz)*causticsOn*lightPower;
  }else{
   float floorT=(-depth-q.y)/min(refr.y,-.0001);
   vec3 floorP=q+refr*max(floorT,0.);
   transmitted=mix(vec3(.10,.20,.22),tiles(floorP.xz)*.6,.4);
   transmitted+=caustic(q.xz+q.y)*.17*causticsOn*lightPower;
  }
  vec3 absorb=exp(-vec3(1.4,.29,.19)*travel);
  transmitted=transmitted*absorb+vec3(.016,.32,.34)*(1.-absorb)*(.55+lightPower*.22);
  float fresnel=.0204+.9796*pow(1.-max(dot(-rd,n),0.),5.);
  vec3 reflected=sky(reflect(rd,n));
  color=mix(transmitted,reflected,clamp(fresnel*reflectionOn,0.,.96));
  vec3 sun=normalize(vec3(-.6,1.,.35));
  float spec=pow(max(dot(reflect(rd,n),sun),0.),210.);
  color+=vec3(1.,.96,.84)*spec*lightPower*reflectionOn*1.8;
  if(side){
   float lip=exp(-abs(p.y-heightAt(p.xz))*95.);
   color+=vec3(.18,.53,.49)*lip;
   float edge=min(abs(abs(p.x)-1.85),abs(abs(p.z)-1.35));
   float corner=max(abs(p.x)/1.85,abs(p.z)/1.35);
   color+=vec3(.03,.09,.09)*smoothstep(.995,1.,corner);
  }
 }
 color=1.-exp(-color*1.25);
 color=pow(color,vec3(.90));
 float vignette=1.-.17*dot(uv,uv);
 gl_FragColor=vec4(color*vignette,1.);
}`;
export function createWater(canvas: HTMLCanvasElement, getSettings: () => WaterSettings, onStats: (fps: number) => void, onError: (message: string) => void) {
 const gl = canvas.getContext('webgl', { alpha: false, antialias: false, powerPreference: 'high-performance' });
 if (!gl) throw new Error('浏览器未能开启 WebGL，请启用硬件加速后重试。');
 const shaders: WebGLShader[] = [];
 const compile = (type: number, source: string) => { const s = gl.createShader(type)!; gl.shaderSource(s, source); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { const error = gl.getShaderInfoLog(s); gl.deleteShader(s); throw new Error(error || '着色器编译失败'); } shaders.push(s); return s; };
 const program = gl.createProgram()!;
 gl.attachShader(program, compile(gl.VERTEX_SHADER, vertex)); gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragment)); gl.linkProgram(program);
 if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || '水体渲染初始化失败');
 gl.useProgram(program);
 const buffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,buffer); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
 const loc=gl.getAttribLocation(program,'position'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
 const uniforms=Object.fromEntries(['resolution','time','amplitude','depth','lightPower','reflectionOn','causticsOn','eye','target','drops[0]'].map(n=>[n,gl.getUniformLocation(program,n)]));
 let yaw=.64, pitch=.53, zoom=7.5,  previous=0, raf=0, frames=0, statTime=0, dropIndex=0, disposed=false;
 const drops=new Float32Array(32); let waveTime=0;
 const camera=()=>[Math.sin(yaw)*Math.cos(pitch)*zoom,Math.sin(pitch)*zoom-.25,Math.cos(yaw)*Math.cos(pitch)*zoom];
 const resize=()=>{const rect=canvas.getBoundingClientRect();const scale=Math.min(window.devicePixelRatio,1.5);canvas.width=Math.max(1,Math.round(rect.width*scale));canvas.height=Math.max(1,Math.round(rect.height*scale));gl.viewport(0,0,canvas.width,canvas.height);};
 const observer=new ResizeObserver(resize);observer.observe(canvas);resize();
 const frame=(now:number)=>{if(disposed)return;const s=getSettings();const dt=previous?Math.min((now-previous)/1000,.05):0;previous=now;if(!s.paused){waveTime+=dt*s.speed;}
 gl.uniform2f(uniforms.resolution,canvas.width,canvas.height);gl.uniform1f(uniforms.time,waveTime);gl.uniform1f(uniforms.amplitude,s.amplitude);gl.uniform1f(uniforms.depth,s.depth);gl.uniform1f(uniforms.lightPower,s.light);gl.uniform1f(uniforms.reflectionOn,+s.reflection);gl.uniform1f(uniforms.causticsOn,+s.caustics);gl.uniform3fv(uniforms.eye,camera());gl.uniform3f(uniforms.target,0,-.32,0);gl.uniform4fv(uniforms['drops[0]'],drops);gl.drawArrays(gl.TRIANGLES,0,6);
 frames++;if(now-statTime>800){onStats(Math.round(frames*1000/(now-statTime)));frames=0;statTime=now;}raf=requestAnimationFrame(frame);};
 const ripple=(x=0,z=0)=>{const i=(dropIndex++%8)*4;drops.set([x,z,waveTime,1],i);};
 let pointer:{id:number;x:number;y:number;lastX:number;lastY:number;moved:boolean}|null=null;
 const down=(e:PointerEvent)=>{pointer={id:e.pointerId,x:e.clientX,y:e.clientY,lastX:e.clientX,lastY:e.clientY,moved:false};canvas.setPointerCapture(e.pointerId);};
 const move=(e:PointerEvent)=>{if(!pointer||pointer.id!==e.pointerId)return;const dx=e.clientX-pointer.lastX,dy=e.clientY-pointer.lastY;if(Math.hypot(e.clientX-pointer.x,e.clientY-pointer.y)>5)pointer.moved=true;if(pointer.moved){yaw-=dx*.007;pitch=Math.max(.15,Math.min(1.3,pitch+dy*.006));}pointer.lastX=e.clientX;pointer.lastY=e.clientY;};
 const normalize=(v:number[])=>{const l=Math.hypot(...v);return v.map(n=>n/l);};const cross=(a:number[],b:number[])=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 const up=(e:PointerEvent)=>{if(!pointer)return;if(!pointer.moved){const rect=canvas.getBoundingClientRect();let u=(e.clientX-rect.left-rect.width*.5)/rect.height;if(rect.width/rect.height>1.18)u+=.19;const v=-(e.clientY-rect.top-rect.height*.5)/rect.height;const eye=camera(),f=normalize([-eye[0],-.32-eye[1],-eye[2]]),r=normalize(cross(f,[0,1,0])),up=cross(r,f);const d=normalize(f.map((n,i)=>n*1.65+u*r[i]+v*up[i]));const t=-eye[1]/d[1],x=eye[0]+d[0]*t,z=eye[2]+d[2]*t;if(t>0&&Math.abs(x)<1.85&&Math.abs(z)<1.35)ripple(x,z);}pointer=null;};
 const cancel=()=>{pointer=null;};const wheel=(e:WheelEvent)=>{e.preventDefault();zoom=Math.max(5.2,Math.min(11,zoom+e.deltaY*.006));};
 const key=(e:KeyboardEvent)=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','Enter',' '].includes(e.key))e.preventDefault();if(e.key==='ArrowLeft')yaw-=.1;if(e.key==='ArrowRight')yaw+=.1;if(e.key==='ArrowUp')pitch=Math.min(1.3,pitch+.08);if(e.key==='ArrowDown')pitch=Math.max(.15,pitch-.08);if(e.key==='+')zoom=Math.max(5.2,zoom-.3);if(e.key==='-')zoom=Math.min(11,zoom+.3);if(e.key==='Enter'||e.key===' ')ripple();};
 const lost=(e:Event)=>{e.preventDefault();cancelAnimationFrame(raf);onError('图形上下文已断开，请点击重新加载恢复水体。');};
 canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointermove',move);canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',cancel);canvas.addEventListener('wheel',wheel,{passive:false});canvas.addEventListener('keydown',key);canvas.addEventListener('webglcontextlost',lost);raf=requestAnimationFrame(frame);
 return {ripple,reset:()=>{yaw=.64;pitch=.53;zoom=7.5;waveTime=0;drops.fill(0);},destroy:()=>{disposed=true;cancelAnimationFrame(raf);observer.disconnect();canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',cancel);canvas.removeEventListener('wheel',wheel);canvas.removeEventListener('keydown',key);canvas.removeEventListener('webglcontextlost',lost);gl.deleteBuffer(buffer);gl.deleteProgram(program);shaders.forEach(s=>gl.deleteShader(s));}};
}
