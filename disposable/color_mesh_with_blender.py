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
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200413_015954/generation_explicit/meshes/02691156/d1a8e79eebf4a0b1579c3d4943e463ef.off'

mesh = trimesh.creation.icosahedron()
P = mesh.vertices.shape[0]
uv = np.zeros([P, 2])
image = Image.new('RGBA', (1, 1), (0, 255, 0, 50))
image.format = 'PNG'
texture = trimesh.visual.TextureVisuals(uv=uv, image=image)
mesh.visual = texture
text, tex_data = trimesh.exchange.obj.export_obj(mesh, include_texture=True)

imagea = Image.new('RGBA', (1, 1), (0, 255, 0, 255))
imagea.format = 'PNG'
imagea.save('temp.png')
imagea = Image.open('temp.png')
imageb = Image.new('RGBA', (1, 1), (255, 0, 0, 255))
imageb.format = 'PNG'
imageb.save('temp.png')
imageb = Image.open('temp.png')

mesha = trimesh.creation.icosahedron()
meshb = mesha.copy()
meshb.vertices -= 0.2
texturea = trimesh.visual.TextureVisuals(uv=uv, image=imagea)
textureb = trimesh.visual.TextureVisuals(uv=uv, image=imageb)
mesha.visual = texturea
meshb.visual = textureb

scene = mesha.scene()
scene.add_geometry(meshb)
text, tex_data = trimesh.exchange.obj.export_obj(scene, include_texture=True)

with open(name, 'w') as f:
    f.write(text)

image = Image.open(io.BytesIO(
    tex_data['material0.png'])).save('disposable/material0.png')

with open('disposable/material0.mtl', 'wb') as f:
    f.write(tex_data['material0.mtl'])

# %%
mesh2 = trimesh.load(name)
mesh2.show()

# %%

# %%
mesh2 = trimesh.load(name)
