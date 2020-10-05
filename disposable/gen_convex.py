# %%
from hull3D import ConvexHull3D
import numpy as np
import trimesh
# %%

pts = np.random.randint(-100, 100, (100, 3))

Hull = ConvexHull3D(pts, run=True, preproc=False, make_frames=False)

# To get Vertex objects:
vertices = list(Hull.DCEL.vertexDict.values())
v_idx = [int(v.identifier) for v in vertices]
verts = np.array([v.p() for v in vertices])
# %%
# To get indices:

# To get vertices of each Face:
faces = np.array([[int(v.identifier) for v in face.loopOuterVertices()]
                  for face in Hull.DCEL.faceDict.values()])
max_int = max(v_idx)
ran = np.arange(max_int + 1)
for idx, idx2 in enumerate(v_idx):
    ran[idx2] = idx
faces2 = np.take(ran, faces.reshape([-1])).reshape([-1, 3])
trimesh.Trimesh(verts, faces2).show()
# %%
