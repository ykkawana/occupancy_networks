from lfd import get_light_field_distance
import trimesh
import tempfile
from bspnet import utils
import numpy as np
import torch

with tempfile.NamedTemporaryFile(
        suffix='.obj') as f1, tempfile.NamedTemporaryFile(
            suffix='.obj') as f2, tempfile.NamedTemporaryFile(
                suffix='.obj') as f3:

    mesh1_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_20200501_185114/meshes/02691156/d1a8e79eebf4a0b1579c3d4943e463ef.off'
    mesh2_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153/meshes/02691156/d1a8e79eebf4a0b1579c3d4943e463ef.off'
    mesh1 = trimesh.load(mesh1_path)
    mesh2 = trimesh.load(mesh2_path)
    mesh1.export(f1.name)
    mesh2.export(f2.name)

    print('test', get_light_field_distance(f1.name, f2.name))
    verts_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet/02691156/d1a8e79eebf4a0b1579c3d4943e463ef/pointcloud.npz'
    verts = np.load(verts_path)['points']
    verts_t = torch.from_numpy(verts).unsqueeze(0).to('cuda').float()

    gt_mesh_path = '/data/ugui0/kawana/ShapeNetCore.v1/02691156/d1a8e79eebf4a0b1579c3d4943e463ef/model.obj'
    gt_mesh = trimesh.load(gt_mesh_path)

    gt_verts = trimesh.util.concatenate(meshes2).vertices
    print(gt_verts.vertices.shape)
    gt_verts_t = torch.from_numpy(gt_verts).to('cuda').float().unsqueeze(0)

    scaled = utils.realign(gt_verts_t, gt_verts_t, verts_t)[0].cpu().numpy()
    gt_mesh.vertices = scaled
    gt_mesh.export(f3.name)

    print('bsp', get_light_field_distance(f1.name, f3.name))
    print('pnet', get_light_field_distance(f2.name, f3.name))
