import yaml
import os
from collections import defaultdict
import copy
import collections.abc
import numpy as np
import subprocess
import trimesh
import os
import plotly.graph_objects as go
import random

from hull3D import ConvexHull3D

side_length_scale = 0.0107337006427915


# %%
def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def isfloat(x):
    try:
        y = float(x)
        return True
    except:
        return False


def isint(x):
    return x.isdigit()


def update_dict_with_options(base, unknown_args):
    print(unknown_args)
    for idx, arg in enumerate(unknown_args):
        if arg.startswith('--'):
            value = unknown_args[idx + 1]
            if value == 'false':
                value = False
            elif value == 'true':
                value = True
            elif value == 'null':
                value = None
            elif value.startswith('[') and value.endswith(']'):
                temp = value[1:-1]
                temp = temp.strip(' ').split(',')
                if temp[0].startswith('"') and temp[0].endswith('"'):
                    value = [s.replace('"', '') for s in temp]
                elif isint(temp[0]):
                    value = map(int, temp)
                elif isfloat(temp[0]):
                    value = map(float, temp)
                else:
                    value = temp
            elif isinstance(value, str) and isint(value):
                value = int(value)
            elif isinstance(value, str) and isfloat(value):
                value = float(value)

            keys = arg.split('.')
            keys[0] = keys[0].replace('--', '')

            print(value, type(value))

            new_dict = {}
            parent_dict = new_dict
            for idx, key in enumerate(keys):
                if idx == len(keys) - 1:
                    parent_dict[key] = value
                else:
                    child_dict = {}
                    parent_dict[key] = child_dict
                    parent_dict = child_dict

            update(base, new_dict)
    return base


def normalize_verts_in_occ_way(vertices):
    vertices = np.copy(vertices)

    mind = [0] * 3
    maxd = [0] * 3

    for i in range(3):
        mind[i] = np.min(vertices[:, i])
        maxd[i] = np.max(vertices[:, i])

    bb_min, bb_max = tuple(mind), tuple(maxd)

    # Get extents of model.
    bb_min, bb_max = np.array(bb_min), np.array(bb_max)
    total_size = (bb_max - bb_min).max()

    # Set the center (although this should usually be the origin already).
    centers = np.array([[(bb_min[0] + bb_max[0]) / 2,
                         (bb_min[1] + bb_max[1]) / 2,
                         (bb_min[2] + bb_max[2]) / 2]])
    # Scales all dimensions equally.
    scale = total_size

    vertices -= centers
    vertices *= 1. / scale
    vertices *= (1 + side_length_scale)

    return vertices


def render_by_blender(rendering_script_path,
                      camera_param_path,
                      model_path,
                      rendering_out_dir,
                      name,
                      skip_reconvert=False,
                      use_cycles=False,
                      use_lamp=False):
    command = 'sh {script} {camera_param} {model} {out_dir} {idx} {skip_reconvert} {use_cycles} {use_lamp}'.format(
        script=rendering_script_path,
        camera_param=camera_param_path,
        model=model_path,
        out_dir=rendering_out_dir,
        idx=name,
        skip_reconvert=('true' if skip_reconvert else 'false'),
        use_cycles=('true' if use_cycles else 'false'),
        use_lamp=('true' if use_lamp else 'false'))
    print(command)
    subprocess.run(command, shell=True)


def export_colored_mesh(meshes, colors, name):
    mtl_format = """
newmtl material{idx}
Ka 0.4 0.4 0.4
Kd {r} {g} {b}
Ks 0.4 0.4 0.4
Ns 1.0
Tf 1.0 1.0 1.0
d {tr}
"""

    scene = meshes[0].scene()
    for mesh in meshes[1:]:
        scene.add_geometry(mesh)
    obj_body_text = trimesh.exchange.obj.export_obj(scene)
    basename = os.path.basename(name)
    basename, _ = os.path.splitext(basename)
    name_wo_ext = os.path.splitext(name)[0]
    obj_body_text = 'mtllib {}.mtl\n'.format(basename) + obj_body_text
    mtl_bodies = []

    assert len(colors) == len(meshes)

    n_verts_sofar = 0
    for idx, color in enumerate(colors):
        color = np.array(color, dtype=np.float32)
        if not np.all(color <= 1) and np.all(color >= 0):
            color /= 255

        obj_body_text = obj_body_text.replace(
            'o geometry_{id}\n'.format(id=idx),
            'o geometry_{id}\nusemtl material{id}\n'.format(id=idx))
        mtl = mtl_format.format(idx=idx,
                                r=color[0],
                                g=color[1],
                                b=color[2],
                                tr=(1.0 if len(color) < 4 else color[3]))
        mtl_bodies.append(mtl)

    with open('{}.mtl'.format(name_wo_ext), 'w') as f:
        for line in mtl_bodies:
            print(line, file=f)

    with open('{}.obj'.format(name_wo_ext), 'w') as f:
        print(obj_body_text, file=f)


def export_textured_mesh(meshes, textures, name):
    mtl_format = """
newmtl material{idx}
Ka 0.4 0.4 0.4
Ks 0.4 0.4 0.4
Ns 1.0
Tf 1.0 1.0 1.0
map_Kd {texture}
"""

    scene = meshes[0].scene()
    for mesh in meshes[1:]:
        scene.add_geometry(mesh)
    obj_body_text = trimesh.exchange.obj.export_obj(scene)

    basename = os.path.basename(name)
    basename, _ = os.path.splitext(basename)
    name_wo_ext = os.path.splitext(name)[0]
    obj_body_text = 'mtllib {}.mtl\n'.format(basename) + obj_body_text
    mtl_bodies = []

    assert len(textures) == len(meshes)

    n_verts_sofar = 0
    for idx, texture in enumerate(textures):

        obj_body_text = obj_body_text.replace(
            'o geometry_{id}\n'.format(id=idx),
            'o geometry_{id}\nusemtl material{id}\n'.format(id=idx))
        mtl = mtl_format.format(idx=idx, texture=texture)
        mtl_bodies.append(mtl)

    with open('{}.mtl'.format(name_wo_ext), 'w') as f:
        for line in mtl_bodies:
            print(line, file=f)

    with open('{}.obj'.format(name_wo_ext), 'w') as f:
        print(obj_body_text, file=f)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
        assert (isinstance(mesh, trimesh.Trimesh))
    return mesh


def export_gt_mesh(shapenet_model_path, export_path):
    mesh = as_mesh(trimesh.load(shapenet_model_path))
    min3 = [0] * 3
    max3 = [0] * 3

    for i in range(3):
        min3[i] = np.min(mesh.vertices[:, i])
        max3[i] = np.max(mesh.vertices[:, i])

    bb_min, bb_max = tuple(min3), tuple(max3)

    # Get extents of model.
    bb_min, bb_max = np.array(bb_min), np.array(bb_max)
    total_size = (bb_max - bb_min).max()

    # Set the center (although this should usually be the origin already).
    centers = np.array([[(bb_min[0] + bb_max[0]) / 2,
                         (bb_min[1] + bb_max[1]) / 2,
                         (bb_min[2] + bb_max[2]) / 2]])
    # Scales all dimensions equally.
    scale = total_size

    mesh.vertices -= centers
    mesh.vertices *= 1. / scale
    mesh.vertices *= (1 + side_length_scale)
    mesh.export(export_path)


def plot_pcd(points, size=1000):
    plots = []
    marker_opt = dict(size=1)

    if isinstance(points, list):
        points_list = points
    else:
        points_list = [points]
    for points in points_list:
        select = np.random.choice(np.arange(len(points)),
                                  size=min(size, len(points)),
                                  replace=False)
        points = points[select, :]
        x2 = points[:, 0]
        y2 = points[:, 1]
        z2 = points[:, 2]
        plots.append(
            go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', marker=marker_opt))

    fig = go.Figure(data=plots)
    fig.update_layout(scene_aspectmode='data')
    fig.show()


def convex_hull(pts):

    Hull = ConvexHull3D(pts, run=True, preproc=False, make_frames=False)

    # To get Vertex objects:
    vertices = list(Hull.DCEL.vertexDict.values())
    v_idx = [int(v.identifier) for v in vertices]
    verts = np.array([v.p() for v in vertices])
    # To get indices:

    # To get vertices of each Face:
    faces = np.array([[int(v.identifier) for v in face.loopOuterVertices()]
                      for face in Hull.DCEL.faceDict.values()])
    max_int = max(v_idx)
    ran = np.arange(max_int + 1)
    for idx, idx2 in enumerate(v_idx):
        ran[idx2] = idx
    faces2 = np.take(ran, faces.reshape([-1])).reshape([-1, 3])
    return trimesh.Trimesh(verts, faces2)


def normals2rgb(normals, append_dummpy_alpha=False):
    normals = normals / np.clip(
        (normals**2).sum(-1, keepdims=True), 1e-7, None)
    RGB = np.zeros_like(normals)
    x = normals[:, 0]
    y = normals[:, 1]
    z = normals[:, 2]
    absx = np.abs(x)
    absy = np.abs(y)
    absz = np.abs(z)
    leftright = (absy >= absx) & (absy >= absz)
    RGB[leftright, 0] = (1 / absy[leftright]) * x[leftright]
    RGB[leftright, 2] = (1 / absy[leftright]) * z[leftright]
    RGB[leftright & (y > 0), 1] = 1
    RGB[leftright & (y < 0), 1] = -1
    frontback = (absx >= absy) & (absx >= absz)
    RGB[frontback, 1] = (1 / absx[frontback]) * y[frontback]
    RGB[frontback, 2] = (1 / absx[frontback]) * z[frontback]
    RGB[frontback & (x > 0), 0] = 1
    RGB[frontback & (x < 0), 0] = -1
    topbottom = (absz >= absx) & (absz >= absy)
    RGB[topbottom, 0] = (1 / absz[topbottom]) * x[topbottom]
    RGB[topbottom, 1] = (1 / absz[topbottom]) * y[topbottom]
    RGB[topbottom & (z > 0), 2] = 1
    RGB[topbottom & (z < 0), 2] = -1
    RGB = 0.5 * RGB + 0.5
    RGB = np.clip(RGB, 0, 1)
    RGB[np.any(np.isnan(RGB), axis=1), :] = [0., 0., 0.]
    if append_dummpy_alpha:
        RGB = np.concatenate(
            [RGB, np.zeros([RGB.shape[0], 1], dtype=RGB.dtype)], axis=1)
    #RGB[np.isnan(normals),2),:)=nan; % zero vectors are mapped to black
    return RGB
