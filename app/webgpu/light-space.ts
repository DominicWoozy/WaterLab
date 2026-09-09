// The existing opaque horizontal receivers. The transparent tank sides remain
// decorative; they are not additional refractive solids.
export const RECEIVER_SIZE = [10, 8] as const;
export const LIGHT_FOOTPRINT = [12, 10] as const;
export const receiverScene = /* wgsl */ `
const RECEIVER_SIZE=vec2f(${RECEIVER_SIZE[0]}.,${RECEIVER_SIZE[1]}.);
const FLOOR_SIZE=vec2f(3.84,2.84);
// Distance and receiver ID: 1=tile platform, 2=surrounding ground, 0=no hit.
fn lightReceiver(ro:vec3f,rd:vec3f)->vec2f {
 if(rd.y>=-.0001){return vec2f(1e5,0.);}
 let floorT=(-.97-ro.y)/rd.y;let p=ro+rd*floorT;
 if(floorT>0.&&all(abs(p.xz)<FLOOR_SIZE*.5)){return vec2f(floorT,1.);}
 let groundT=(-1.075-ro.y)/rd.y;
 if(groundT>0.){return vec2f(groundT,2.);}return vec2f(1e5,0.);
}
`;
