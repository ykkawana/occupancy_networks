# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import kaolin as kal
import json
import os
import glob
import scipy.io
import trimesh
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import kaolin.conversions.meshconversions as mesh_cvt
from kaolin.transforms import pointcloudfunc as pcfunc
from kaolin.transforms import meshfunc

from kaolin.transforms import transforms as tfs
import sys
sys.path.insert(0, '/home/mil/kawana/workspace/occupancy_networks')
from im2mesh.utils import binvox_rw
import json


# %%
def join(*args):
    return os.path.join(*args)


def rotz(d):
    d = d / 180. * np.pi
    return np.array([[1., 0., 0.], [0., np.cos(d), -np.sin(d)],
                     [0., np.sin(d), np.cos(d)]])


def rotx(d):
    d = d / 180. * np.pi

    return np.array([[np.cos(d), 0., np.sin(d)], [0., 1., 0.],
                     [-np.sin(d), 0., np.cos(d)]])


def roty(d):
    d = d / 180. * np.pi

    return np.array([[np.cos(d), -np.sin(d), 0.], [np.sin(d),
                                                   np.cos(d), 0.],
                     [0., 0., 1.]])


def rot(m, x, y, z):
    rotmat = np.dot(rotz(z), np.dot(roty(y), rotx(x)))
    rotmatt = torch.tensor(rotmat, device=m.device).float()
    return meshfunc.rotate(m, rotmat=rotmatt, inplace=True)


# %%
def convert_and_save_models(voxel_path, output_path):
    voxel = scipy.io.loadmat(voxel_path)['voxel']
    voxel_tensor = torch.tensor(voxel).to('cuda')

    mesh_conversion = tfs.VoxelGridToTriangleMesh(threshold=0.5,
                                                  mode='marching_cubes',
                                                  normalize=True)

    transforms = tfs.Compose(
        [mesh_conversion,
         tfs.MeshLaplacianSmoothing(smoothing_iterations)])

    mesh = transforms(voxel_tensor)

    rot(mesh, 0, -90, -90)

    mesh.vertices *= (1 + side_length_scale)  # to adjust for occnet box size

    sdf = mesh_cvt.trianglemesh_to_sdf(mesh, num_points)
    bbox_true = torch.stack(
        (mesh.vertices.min(dim=0)[0], mesh.vertices.max(dim=0)[0]),
        dim=1).view(-1)
    points = 1.05 * (torch.rand(num_points, 3).to(mesh.vertices.device) - .5)
    distances = sdf(points)
    occupancies = distances <= 0

    pcd_points, face_choices = mesh_cvt.trianglemesh_to_pointcloud(
        mesh, num_points)
    face_normals = mesh.compute_face_normals()
    pcd_point_normals = face_normals[face_choices]

    voxel_binvox = binvox_rw.Voxels(voxel, voxel.shape, [0., 0., 0.], 1.,
                                    'xyz')

    npv, npf = mesh.vertices.to('cpu').numpy(), mesh.faces.to('cpu').numpy()
    np_mesh = trimesh.Trimesh(npv, npf)
    hashd = np_mesh.md5()
    model_output_path = join(output_path, hashd)
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    np_mesh.export(join(model_output_path, 'model.off'))
    binvox_rw.write(voxel_binvox,
                    open(join(model_output_path, 'model.binvox'), 'w'))

    packed_occupancies = np.packbits(occupancies.to('cpu').numpy())
    np_points = points.float().to('cpu').numpy()
    np.savez(join(model_output_path, 'points.npz'),
             occupancies=packed_occupancies,
             points=np_points,
             side_length_scale=side_length_scale)

    np_pcd_points = pcd_points.float().to('cpu').numpy()
    np_pcd_point_normals = pcd_point_normals.float().to('cpu').numpy()
    np.savez(join(model_output_path, 'pointcloud.npz'),
             points=np_pcd_points,
             normals=np_pcd_point_normals,
             side_length_scale=side_length_scale)

    np_distances = distances.float().to('cpu').numpy()
    np.savez(join(model_output_path, 'sdf_points.npz'),
             points=np_pcd_points,
             distances=np_distances,
             side_length_scale=side_length_scale)

    return hashd


# %%
divide_all = int(sys.argv[1])
divide_id = int(sys.argv[2])
assert divide_all > divide_id
convert_model = True

num_points = 100000
smoothing_iterations = 3
side_length_scale = 0.0107337006427915
output_base_path = '/data/ugui0/kawana/ShapeNetLikePix3D_correct_direction'
pix3d_base_path = '/data/ugui0/kawana/pix3d'
pix3d_json_path = join(pix3d_base_path, 'pix3d.json')
#pix3d_json_path = '/home/mil/kawana/workspace/occupancy_networks/minipix3d.json'

synset_to_label = {
    '04256520': 'sofa',
    '04379243': 'table',
    '02691156': 'bed',
    '02828884': 'bookcase',
    '02933112': 'desk',
    '02958343': 'misc',
    '03001627': 'chair',
    '03211117': 'tool',
    '03636649': 'wardrobe'
}

label_to_synset = {v: k for k, v in synset_to_label.items()}

# %%
metadata = {
    synset: {
        "id": synset,
        "name": label
    }
    for synset, label in synset_to_label.items()
}
json.dump(metadata,
          open(join(output_base_path, 'metadata.yaml'), 'w'),
          ensure_ascii=False,
          indent=4,
          sort_keys=True,
          separators=(',', ': '))

# %%
for synset in synset_to_label:
    class_path = join(output_base_path, synset)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

# %%
print(divide_id, divide_all)
pix3d_dicts = json.load(open(pix3d_json_path))
pix3d_dicts = np.array_split(pix3d_dicts, divide_all)[divide_id].tolist()
print(len(pix3d_dicts))
model_n = len(pix3d_dicts)
print('total:', model_n)
new_pix3d_dicts = []

pbar = tqdm(total=model_n)
for model_dict in pix3d_dicts:
    synset = label_to_synset[model_dict['category']]
    voxel_path = join(pix3d_base_path, model_dict['voxel'])
    output_path = join(output_base_path, synset)
    #try:
    modelname = convert_and_save_models(voxel_path, output_path)
    model_dict['class_id'] = synset
    model_dict['modelname'] = modelname
    model_dict['class_name'] = synset_to_label[synset]
    #except:
    #    print('fail to convert:', voxel_path)

    new_pix3d_dicts.append(model_dict)
    pbar.update(1)

df = pd.DataFrame(new_pix3d_dicts)
pickle.dump(
    df,
    open(
        join(output_base_path, 'pix3d_{}_{}.pkl'.format(divide_id,
                                                        divide_all)), 'wb'))
