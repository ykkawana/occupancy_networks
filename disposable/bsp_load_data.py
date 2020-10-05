# %%
import trimesh
import eval_utils
import os
import numpy as np
import h5py
import glob
import torch
from kaolin.transforms import pointcloudfunc as pcfunc
#%%
"""
path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNetBSPNet/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/pointcloud.ply'

mesh = trimesh.load(path)

#eval_utils.plot_pcd(mesh.vertices)

# %%

target = '02691156/d1cdd239dcbfd018bbf3143b1cb6076a'
target = '03001627/d2c465e85d2e8f1fcea003eff0268278'

"""


def get_bsp_data(target):
    sample_vox_size = 64
    if sample_vox_size == 16:
        load_point_batch_size = 16 * 16 * 16
    elif sample_vox_size == 32:
        load_point_batch_size = 16 * 16 * 16
    elif sample_vox_size == 64:
        load_point_batch_size = 16 * 16 * 16 * 4
    shape_batch_size = 24
    point_batch_size = 16 * 16 * 16
    input_size = 64  #input voxel grid size

    ef_dim = 32
    p_dim = 4096

    dataset_name = 'all_vox256_img'
    data_dir = '/home/mil/kawana/workspace/occupancy_networks/external/bspnet/data/all_vox256_img'
    point_cloud_dir = ''
    lines = []
    class_id = target.split('/')[0]
    modelname = target.split('/')[1]
    target_suffix = ''
    pointcloud_dir = '/data/ugui0/kawana/ShapeNetBAENet/data/'
    pointcloud_path = glob.glob(pointcloud_dir + class_id + '*')[0]
    pointcloud = np.loadtxt(pointcloud_path + '/points/' + modelname +
                            '.txt')[:, :3]
    for suffix in ['_test', '_train']:
        dataset_load = dataset_name + suffix
        data_txt_name = data_dir + '/' + dataset_load + '.txt'

        with open(data_txt_name) as f:
            lines = [l.strip() for l in f.readlines()]

        if target in lines:
            idx = lines.index(target)
            target_suffix = suffix
            break

    dataset_load = dataset_name + target_suffix
    data_hdf5_name = data_dir + '/' + dataset_load + '.hdf5'
    if os.path.exists(data_hdf5_name):
        data_dict = h5py.File(data_hdf5_name, 'r')
        data_points = (data_dict['points_' + str(sample_vox_size)][idx].astype(
            np.float32) + 0.5) / 256 - 0.5

        data_values = data_dict['values_' + str(sample_vox_size)][idx].astype(
            np.float32)
        data_voxels = data_dict['voxels'][idx]
        #reshape to NCHW
        data_voxels = np.reshape(data_voxels,
                                 [-1, 1, input_size, input_size, input_size])
    else:
        print("error: cannot load " + data_hdf5_name)
        exit(0)

    #%%

    rad = np.pi / 2.
    roty = [[np.cos(rad), 0., np.sin(rad)], [0., 1., 0.],
            [-np.sin(rad), 0., np.cos(rad)]]

    rotm = torch.tensor(roty, device='cuda').float()

    invrad = -np.pi / 2.
    invroty = [[np.cos(invrad), 0., np.sin(invrad)], [0., 1., 0.],
               [-np.sin(invrad), 0., np.cos(invrad)]]

    inv_rotm = torch.tensor(invroty, device='cuda').float()

    def roty90(points_iou, inv=False):
        if inv:
            return pcfunc.rotate(points_iou, inv_rotm)
        else:
            return pcfunc.rotate(points_iou, rotm)

    # %%
    return roty90(torch.from_numpy(data_points).to('cuda'),
                  inv=True).cpu().numpy().astype(
                      np.float32), data_values.astype(
                          np.float32), pointcloud.astype(np.float32)


#  %%
"""
rotated = data_points[data_values.reshape(-1) == 1, :]
rotated = roty90(torch.from_numpy(rotated).to('cuda'), inv=True).cpu().numpy()
pointclouds = [rotated, pointcloud]
eval_utils.plot_pcd(pointclouds, size=5000)

# %%
data_dir = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
pointcloud_path = os.path.join(data_dir, target, 'pointcloud.npz')
points_path = os.path.join(data_dir, target, 'points.npz')

pointcloud = np.load(pointcloud_path)['points']
points = np.load(points_path)['points']
values = np.unpackbits(np.load(points_path)['occupancies'])

pointclouds = [pointcloud, points[values == 1, :]]
eval_utils.plot_pcd(pointclouds, size=5000)

# %%

"""
