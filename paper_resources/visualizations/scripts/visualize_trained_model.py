# %%
import plotly.graph_objects as go
import trimesh
import numpy as np
from mayavi import mlab
mlab.init_notebook()

# %%
path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNetSDF/02691156/1a04e3eab45ca15dd86060f189eb133/sdf_points.npz'
points = np.load(path)['points']
sdf = np.load(path)['distances']
inside = points[sdf <= 0.00001, :]
x = inside[:, 0]
y = inside[:, 1]
z = inside[:, 2]
outside = points[(0.1 > sdf) & (sdf > 0.01), :]
x2 = outside[:, 0]
y2 = outside[:, 1]
z2 = outside[:, 2]
plots = [
    go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1)),
    go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', marker=dict(size=1))
]

layout = go.Layout(xaxis=go.XAxis(showticklabels=False),
                   yaxis=go.YAxis(showticklabels=False),
                   scene_aspectmode='data',
                   xaxis_showgrid=False,
                   yaxis_showgrid=False)
fig = go.Figure(data=plots)
fig.update_layout(layout)
fig.show()
# %%
pcd = trimesh.PointCloud(points)
scene = trimesh.Scene(pcd)
scene.show()
