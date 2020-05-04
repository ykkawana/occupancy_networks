# %%
from PIL import Image
import os
os.chdir('/home/mil/kawana/workspace/occupancy_networks')
import eval_utils
import trimesh
import numpy as np
import io

# %%
rendering_out = '/home/mil/kawana/workspace/occupancy_networks/disposable'
camera_param = 'paper_resources/compare_mesh_methods/camera_param.txt'
render_script_path = '/home/mil/kawana/workspace/occupancy_networks/scripts/render_3dobj.sh'
name = 'disposable/temp.obj'

# %%

colors = [[0, 1, 0], [1, 0, 0]]


def export_colored_mesh(meshes, colors, name):
    mtl_format = """
    newmtl material{idx}
    Ka 0.4 0.4 0.4
    Kd {r} {g} {b}
    Ks 0.4 0.4 0.4
    Ns 1.0
    """

    scene = meshes[0].scene()
    for mesh in meshes[1:]:
        scene.add_geometry(mesh)
    obj_body_text = trimesh.exchange.obj.export_obj(scene)

    basename = os.path.basename(name)
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
        mtl = mtl_format.format(idx=idx, r=color[0], g=color[1], b=color[2])
        mtl_bodies.append(mtl)

    with open('{}.mtl'.format(name), 'w') as f:
        for line in mtl_bodies:
            print(line, file=f)

    with open('{}.obj'.format(name), 'w') as f:
        print(obj_body_text, file=f)


mesha = trimesh.creation.icosahedron()
meshb = mesha.copy()
meshb.vertices -= 0.2
name = 'disposable/color'
export_colored_mesh([mesha, meshb], [[255, 0, 0], [0, 0, 255]], name)
# %%
mesh2 = trimesh.load(name + '.obj')
mesh2.show()

# %%
eval_utils.render_by_blender(render_script_path, camera_param, name + '.obj',
                             rendering_out, 'color')
