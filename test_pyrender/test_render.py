# %%
# Render offscreen -- make sure to set the PyOpenGL platform
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender
import tempfile
import matplotlib.pyplot as plt
import time
# %%

r = pyrender.OffscreenRenderer(640, 480)


# %%
def resize(ratio):
    return np.array([
        [ratio, 0, 0, 0],
        [0, ratio, 0.0, 0],
        [0, 0, ratio, 0],
        [0.0, 0.0, 0.0, 1],
    ])


def translate(x=0, y=0, z=0):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])


def rotate(x=0, y=0, z=0):
    xrd = x / 180 * np.pi
    yrd = y / 180 * np.pi
    zrd = z / 180 * np.pi

    xr = np.array([
        [1, 0, 0, 0],
        [0, np.cos(xrd), -np.sin(xrd), 0],
        [0, np.sin(xrd), np.cos(xrd), 0],
        [0, 0, 0, 1],
    ])

    yr = np.array([
        [np.cos(yrd), 0, np.sin(yrd), 0],
        [0, 1, 0, 0],
        [-np.sin(yrd), 0, np.cos(yrd), 0],
        [0, 0, 0, 1],
    ])

    zr = np.array([[np.cos(zrd), -np.sin(zrd), 0, 0],
                   [np.sin(zrd), np.cos(zrd), 0, 0], [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return np.dot(zr, np.dot(yr, xr))


def transform(trans):
    res = trans[0]
    for idx in range(len(trans) - 1):
        res = np.dot(trans[idx + 1], res)
    return res


#%%
# Load the FUZE bottle trimesh and put it in a scene
fuze_trimesh = trimesh.load(
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/meshes/02691156/d1a8e79eebf4a0b1579c3d4943e463ef.off'
)
#fuze_trimesh = trimesh.load('/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403/meshes/02691156/d1a887a47991d1b3bc0909d98a1ff2b4.off')
with tempfile.NamedTemporaryFile(suffix='.obj') as f:
    fuze_trimesh.export(f.name)
    fuze_trimesh = trimesh.load(f.name)
#fuze_trimesh = trimesh.load('pyrender/examples/models/fuze.obj')
#fuze_trimesh = trimesh.load('/data/ugui0/kawana/ShapeNetCore.v1/02691156/1ea8a685cdc71effb8494b55ada518dc/model.obj')
fuze_trimesh.apply_transform(
    transform([rotate(x=30, y=-30, z=00),
               translate(z=-.9)]))
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh,
                                  material=pyrender.MetallicRoughnessMaterial(
                                      baseColorFactor=[.4, .4, .4, 1.]),
                                  smooth=False)
scene = pyrender.Scene()
#scene = pyrender.Scene(ambient_light=[1., 1., 1.])
scene.add(mesh)

# Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
s = np.sqrt(2) / 2
camera_pose = np.array([
    [1.0, -0, 0, .0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)

# Set up the light -- a single spot light in the same spot as the camera
#light = pyrender.SpotLight(color=np.ones(3),
#                           intensity=3.0,
#                           innerConeAngle=np.pi / 16.0)
scene.add(light, pose=camera_pose)

# Render the scene
flags = pyrender.constants.RenderFlags.OFFSCREEN
t0 = time.time()
for _ in range(100):
    color, depth = r.render(scene, flags)
    # Show the images
    plt.figure()
    #plt.figure(figsize=(50,50))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    #plt.imshow(color)
    plt.savefig('test.png')
print('total', time.time() - t0)
"""
plt.subplot(1, 2, 2)
plt.axis('off')
plt.imshow(depth, cmap=plt.cm.gray_r)
"""
#plt.show()
