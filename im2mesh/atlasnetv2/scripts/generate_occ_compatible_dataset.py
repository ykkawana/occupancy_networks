# %%
import os
import glob
import multiprocessing as mp
import trimesh
import numpy as np
import torch
from bspnet.utils import realign
from tqdm import tqdm

from joblib import Parallel, delayed
from kaolin.transforms import pointcloudfunc as pcfunc
import pandas as pd
# %%
os.chdir('/home/mil/kawana/workspace/occupancy_networks')
# %%
synset_to_label = {
    "02691156": "airplane",
    "02828884": "bench",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03211117": "display",
    "03636649": "lamp",
    "03691459": "loudspeaker",
    "04090263": "rifle",
    "04256520": "sofa",
    "04379243": "table",
    "04401088": "telephone",
    "04530566": "vessel"
}
label_to_synset = {v: k for k, v in synset_to_label.items()}

atv2_root = 'data/ShapeNetAtlasNetV2'
atv2_coverted_root = 'data/ShapeNetAtlasNetV2Converted'
occ_root = 'data/ShapeNet'


# %%
def convert(inputs):
    device = 'cuda'
    rad = np.pi / 2.
    roty = [[np.cos(rad), 0., np.sin(rad)], [0., 1., 0.],
            [-np.sin(rad), 0., np.cos(rad)]]

    rotm = torch.tensor(roty, device=device).float()

    invrad = -np.pi / 2.
    invroty = [[np.cos(invrad), 0., np.sin(invrad)], [0., 1., 0.],
               [-np.sin(invrad), 0., np.cos(invrad)]]

    inv_rotm = torch.tensor(invroty, device=device).float()

    def roty90(points_iou, inv=False):
        if inv:
            return pcfunc.rotate(points_iou, inv_rotm)
        else:
            return pcfunc.rotate(points_iou, rotm)

    print('start parallel')
    indices, dataset, atv2_root, atv2_coverted_root, occ_root = inputs

    for it in tqdm(indices):
        #if it > 10:
        #    break

        class_id, modelname = dataset[it]

        if not os.path.exists(
                os.path.join(atv2_coverted_root, class_id, modelname)):
            os.makedirs(os.path.join(atv2_coverted_root, class_id, modelname))

        atv2ply_path = os.path.join(atv2_root, class_id, 'ply',
                                    modelname + '.points.ply')
        occpoints_path = os.path.join(occ_root, class_id, modelname,
                                      'pointcloud.npz')

        out_path = os.path.join(atv2_coverted_root, class_id, modelname,
                                'pointcloud')

        if os.path.exists(out_path + '.npz'):
            continue

        if not os.path.exists(atv2ply_path) or not os.path.exists(
                occpoints_path):
            continue

        try:
            atv2mesh = trimesh.load(atv2ply_path)
            occpoints = np.load(occpoints_path)['points']

            atv2points = torch.from_numpy(atv2mesh.vertices /
                                          2).float().unsqueeze(0).to('cuda')
            occpoints = torch.from_numpy(occpoints).float().unsqueeze(0).to(
                'cuda')

            atv2points = roty90(atv2points, inv=True)

            atv2points = realign(atv2points,
                                 atv2points,
                                 occpoints,
                                 adjust_bbox=False)

            np.savez(out_path, points=atv2points.detach().cpu()[0].numpy())
        except:
            continue


# %%
num_proc = 12
occ_list = []
for class_id in synset_to_label:
    occ_list.extend([
        (class_id, modelname)
        for modelname in os.listdir(os.path.join(occ_root, class_id))
        if os.path.isdir(os.path.join(occ_root, class_id, modelname))
    ])

occ_set = set(occ_list)
indices = np.random.permutation(np.arange(len(occ_set)))
indices_list = np.array_split(indices, num_proc)
"""
convert((indices, atv2_list, atv2_root, atv2_coverted_root, occ_root))
"""
r = Parallel(n_jobs=num_proc)([
    delayed(convert)(
        (indices, occ_list, atv2_root, atv2_coverted_root, occ_root))
    for indices in indices_list
])
# %%

atv2_list = []
for class_id in synset_to_label:
    atv2_list.extend([(class_id, path.split('/')[-2]) for path in glob.glob(
        os.path.join(atv2_coverted_root, class_id, '*/pointcloud.npz'))])

atv2_set = set(atv2_list)

print(len(atv2_list), len(occ_list))
# %%
for class_id in synset_to_label:
    if not os.path.exists(os.path.join(atv2_coverted_root, class_id)):
        os.makedirs(os.path.join(atv2_coverted_root, class_id))

    train_list = []
    test_list = []
    for txt in ['train.lst', 'val.lst', 'test.lst']:
        path = os.path.join(occ_root, class_id, txt)
        with open(path) as f:
            modelnames = [(class_id, line.strip()) for line in f.readlines()]
        inter = set(modelnames).intersection(atv2_set)
        modelnames = [modelname for class_id, modelname in inter]
        if txt in ['train.lst', 'val.lst']:
            train_list.extend(modelnames)
        else:
            test_list = modelnames
    with open(os.path.join(atv2_coverted_root, class_id, 'train.lst'),
              'w') as f:
        modelnames_txt = '\n'.join(train_list)
        f.write(modelnames_txt)
    with open(os.path.join(atv2_coverted_root, class_id, 'test.lst'),
              'w') as f:
        modelnames_txt = '\n'.join(test_list)
        f.write(modelnames_txt)

indices = np.arange(len(atv2_list))
