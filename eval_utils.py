import yaml
import os
from collections import defaultdict
import copy
import collections.abc
import numpy as np
import subprocess
import trimesh
import os

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
                      skip_reconvert=False):
    command = 'sh {script} {camera_param} {model} {out_dir} {idx} {skip_reconvert}'.format(
        script=rendering_script_path,
        camera_param=camera_param_path,
        model=model_path,
        out_dir=rendering_out_dir,
        idx=name,
        skip_reconvert=('true' if skip_reconvert else 'false'))
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
