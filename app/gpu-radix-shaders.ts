import { PARTICLE_WIDTH } from './gpu-particle-config.ts';

// Stable LSD radix-16 sort. Each block holds 32 keys; four RGBA texels hold
// its 16 bin counts. Prefix sums assign exact, collision-free scatter slots.
const header = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D sortedKeys, radixRanks, radixHistogram;
uniform int sortCount, digitShift, scanStride;
ivec2 uv(int i){return ivec2(i%${PARTICLE_WIDTH},i/${PARTICLE_WIDTH});}
vec4 at(sampler2D t,int i){return texelFetch(t,uv(i),0);}
// Valid cell keys are 0..30719. Padding sorts after every live particle.
int digit(float key){return (int(min(key,65535.))>>digitShift)&15;}
`;

export const radixRankFragment =
  header +
  `
out vec4 result;
void main(){
 int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*${PARTICLE_WIDTH};
 vec4 a=at(sortedKeys,i);int d=digit(a.x),base=i&~31,rank=0;
 for(int j=0;j<32;j++){
  if(j>=(i&31))break;
  if(digit(at(sortedKeys,base+j).x)==d)rank++;
 }
 result=vec4(a.xy,float(rank),float(d));
}`;

export const radixHistogramFragment =
  header +
  `
out vec4 result;
void main(){
 int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*${PARTICLE_WIDTH};
 int base=(i/4)*32;ivec4 bins=ivec4(0,1,2,3)+(i&3)*4;
 vec4 counts=vec4(0.);
 for(int j=0;j<32;j++){
  int d=digit(at(sortedKeys,base+j).x);
  counts+=vec4(equal(ivec4(d),bins));
 }
 result=counts;
}`;

export const radixScanFragment =
  header +
  `
out vec4 result;
void main(){
 int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*${PARTICLE_WIDTH};
 int block=i/4;vec4 sum=at(radixHistogram,i);
 // Each stage concatenates 16 non-overlapping intervals from the prior stage.
 for(int j=1;j<16;j++){
  if(block<j*scanStride)break;
  sum+=at(radixHistogram,i-j*scanStride*4);
 }
 result=sum;
}`;

export const radixScatterVertex =
  header +
  `
flat out vec2 keyPair;
void main(){
 int i=gl_VertexID;vec4 a=at(radixRanks,i);
 int d=int(a.w),pack=d/4,component=d&3,block=i/32;
 int last=(sortCount/32-1)*4;float offset=0.;
 for(int p=0;p<4;p++){
  if(p>pack)break;
  vec4 totals=at(radixHistogram,last+p);
  for(int c=0;c<4;c++)if(p*4+c<d)offset+=totals[c];
 }
 if(block>0)offset+=at(radixHistogram,(block-1)*4+pack)[component];
 int dest=int(offset+a.z);
 vec2 pixel=vec2(uv(dest))+.5;
 gl_Position=vec4(pixel/vec2(${PARTICLE_WIDTH}.,float(sortCount/${PARTICLE_WIDTH}))*2.-1.,0.,1.);
 gl_PointSize=1.;keyPair=a.xy;
}`;

export const radixScatterFragment = `#version 300 es
precision highp float;
flat in vec2 keyPair;
out vec4 result;
void main(){result=vec4(keyPair,0.,0.);}`;
