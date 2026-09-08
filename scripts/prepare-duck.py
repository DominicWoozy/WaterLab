"""Repack the original Sony/Khronos mesh into a GPU ray-tracing BVH.
No generated geometry, subdivision, decimation, normal or texture changes.
Uniform scale / translation align the supplied model with the rigid body's COM.
"""
import json,struct,math,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];folder=root/'public/models/duck'
data=(folder/'Duck.glb').read_bytes();assert data[:4]==b'glTF'
length,typ=struct.unpack_from('<II',data,12);doc=json.loads(data[20:20+length]);offset=20+length
length,typ=struct.unpack_from('<II',data,offset);blob=data[offset+8:offset+8+length]
def accessor(i):
 a=doc['accessors'][i];v=doc['bufferViews'][a['bufferView']];n={'SCALAR':1,'VEC2':2,'VEC3':3}[a['type']];fmt={5123:'H',5126:'f'}[a['componentType']];stride=v.get('byteStride',struct.calcsize(fmt)*n);start=v.get('byteOffset',0)+a.get('byteOffset',0)
 return [struct.unpack_from('<'+fmt*n,blob,start+j*stride) for j in range(a['count'])]
primitive=doc['meshes'][0]['primitives'][0];positions=accessor(primitive['attributes']['POSITION']);normals=accessor(primitive['attributes']['NORMAL']);uvs=accessor(primitive['attributes']['TEXCOORD_0']);indices=[x[0] for x in accessor(primitive['indices'])]
positions=[(p[0]*.0045,(p[1]-9.929369926452637)*.0045-.14,(p[2]+3.7015)*.0045) for p in positions]
triangles=[indices[i:i+3] for i in range(0,len(indices),3)];nodes=[];packed=[]
def build(tris,depth=0):
 lo=[min(positions[i][a] for tri in tris for i in tri)-1e-5 for a in range(3)];hi=[max(positions[i][a] for tri in tris for i in tri)+1e-5 for a in range(3)];idx=len(nodes);nodes.append(None)
 if len(tris)<=6:
  start=len(packed);packed.extend(tris);nodes[idx]=lo+[-float(start)-1]+hi+[float(len(tris))]
 else:
  axis=max(range(3),key=lambda a:hi[a]-lo[a]);tris.sort(key=lambda t:sum(positions[i][axis] for i in t));mid=len(tris)//2;left=build(tris[:mid],depth+1);right=build(tris[mid:],depth+1);nodes[idx]=lo+[float(left)]+hi+[float(right)]
 return idx
build(triangles)
texels=[]
for tri in packed:
 for i in tri:texels.extend([*positions[i],uvs[i][0]])
 for i in tri:texels.extend([*normals[i],uvs[i][1]])
def write(name,values):
 height=math.ceil(len(values)/4/256);values+= [0.]*(height*256*4-len(values));(folder/name).write_bytes(struct.pack('<'+'f'*len(values),*values));return height
meta={'triangles':len(triangles),'vertices':len(positions),'nodes':len(nodes),'width':256,'triangleHeight':write('triangles.bin',texels),'bvhHeight':write('bvh.bin',[v for n in nodes for v in n]),'sourceSha256':hashlib.sha256(data).hexdigest(),'bounds':[nodes[0][:3],nodes[0][4:7]]}
(folder/'mesh.json').write_text(json.dumps(meta,indent=2)+'\n')
v=doc['bufferViews'][doc['images'][0]['bufferView']];(folder/'DuckCM.png').write_bytes(blob[v.get('byteOffset',0):v.get('byteOffset',0)+v['byteLength']])
print(json.dumps(meta,indent=2))
for lo,hi in [(-.14,.05),(.05,.18),(.18,.3),(.3,.42),(.42,.55)]:
 points=[p for p in positions if lo<p[1]<hi];print('slice',lo,hi,[(round(min(p[a] for p in points),3),round(max(p[a] for p in points),3)) for a in [0,2]])
