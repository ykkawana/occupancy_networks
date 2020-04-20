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
from kaolin.transforms import transforms as tfs
import sys
sys.path.insert(0, '/home/mil/kawana/workspace/occupancy_networks')
from im2mesh.utils import binvox_rw
import json
import skimage.io
import skimage.transform

is_jupyter = True


# %%
def join(*args):
    return os.path.join(*args)


# %%
if is_jupyter:
    divide_all = 1
    divide_id = 0
else:
    divide_all = int(sys.argv[1])
    divide_id = int(sys.argv[2])
assert divide_all > divide_id
convert_model = True

num_points = 100000
smoothing_iterations = 3
side_length_scale = 0.0107337006427915
out_image_size = np.array([224, 224])
#output_base_path = '/data/ugui0/kawana/ShapeNetLikePix3D2'
output_base_path = '/data/ugui0/kawana/ShapeNetLikePix3D_correct_direction'
pix3d_base_path = '/data/ugui0/kawana/pix3d'
pix3d_df_path = join(output_base_path, 'pix3d_*.pkl')
choy2016margin_df_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/pix3d_comparison/margin.pkl'
output_image_dir_name = 'class_agnostic_margin_224'
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

common_class_indices = [0, 1, 6]
uncommon_class_indices = list(
    set(range(len(synset_to_label))) - set(common_class_indices))
common_class_synsets = [
    key for idx, key in enumerate(synset_to_label.keys())
    if idx in common_class_indices
]
uncommon_class_synsets = [
    key for idx, key in enumerate(synset_to_label.keys())
    if idx not in common_class_indices
]

# %%
print(divide_id, divide_all)
pix3d_df_paths = glob.glob(pix3d_df_path)
dfs = []
for path in pix3d_df_paths:
    pix3d_df = pickle.load(open(path, 'rb'))
    dfs.append(pix3d_df)
pix3ddf = pd.concat(dfs)
margindf = pickle.load(open(choy2016margin_df_path, 'rb'))
# %%
margin_mean = margindf.groupby('class_id').mean().mean()
margin_std = margindf.groupby('class_id').std().mean()

unoises = np.random.normal(loc=margin_mean['u'].item() * out_image_size[0],
                           scale=margin_std['u'].item(),
                           size=len(pix3ddf) + 100)
bnoises = np.random.normal(loc=margin_mean['b'].item() * out_image_size[0],
                           scale=margin_std['b'].item(),
                           size=len(pix3ddf) + 100)
lnoises = np.random.normal(loc=margin_mean['l'][0].item() * out_image_size[1],
                           scale=margin_std['l'][0].item(),
                           size=len(pix3ddf) + 100)
rnoises = np.random.normal(loc=margin_mean['r'].item() * out_image_size[1],
                           scale=margin_std['r'].item(),
                           size=len(pix3ddf) + 100)

owidth = margin_mean['w'].item()
oheight = owidth

scale = out_image_size[0] / owidth
pbar = tqdm(total=len(pix3ddf))
samples_per_model = {}
for idx, item in pix3ddf.iterrows():
    img_relative_path = item['img']
    mask_relative_path = item['mask']
    class_id = item['class_id']
    modelname = item['modelname']
    if modelname not in samples_per_model:
        samples_per_model[modelname] = [
            0, len(pix3ddf[pix3ddf['modelname'] == modelname])
        ]
    cnt = samples_per_model[modelname][0]
    samples_per_model[modelname][0] += 1

    img_path = join(pix3d_base_path, img_relative_path)
    mask_path = join(pix3d_base_path, mask_relative_path)

    try:
        img = skimage.io.imread(img_path)
    except:
        print('read image error', img_path)
        continue
    if img.ndim == 3:
        img = img[:, :, :3]
    elif img.ndim == 2:
        img = np.tile(img[..., np.newaxis], (1, 1, 3))
    else:
        print('unsupported image dimension')
        continue
    mask = skimage.io.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 1]

    if not mask.shape[:2] == img.shape[:2]:
        print('mask image mismtach', img_path)
        continue

    yidx, xidx = np.where(mask == 255)
    h, w = mask.shape
    u = yidx.min()
    b = yidx.max()
    l = xidx.min()
    r = xidx.max()

    cropped = (img * (mask[..., np.newaxis] == 255))[u:b, l:r, :]
    back = np.ones_like(cropped) * 255
    invmask = (mask[..., np.newaxis] != 255)[u:b, l:r, :]
    back *= invmask
    #back[mask[..., np.newaxis][u:b, l:r, :], :] = 0
    cropped += back
    ch, cw, _ = cropped.shape
    if ch > cw:
        sm = unoises[idx]
        iidx = 0
    else:
        sm = lnoises[idx]
        iidx = 1
    new_length = out_image_size[iidx] - int(sm * 2)
    if new_length >= out_image_size[iidx]:
        new_length = out_image_size[iidx]
        sm = 0
    if ch > cw:
        new_scale = new_length / ch
    else:
        new_scale = new_length / cw

    resized_cropped = skimage.transform.rescale(cropped,
                                                [new_scale, new_scale, 1],
                                                preserve_range=True)

    rch, rcw, _ = resized_cropped.shape
    sm_int = int(sm)

    new_image = np.ones([*out_image_size, 3], dtype=np.uint8) * 255

    if ch > cw:
        wp = int((out_image_size[1] - rcw) / 2)
        new_image[sm_int:(sm_int + rch), wp:(wp + rcw), :] = resized_cropped[:]
    else:
        hp = int((out_image_size[0] - rch) / 2)
        new_image[hp:(hp + rch), sm_int:(sm_int + rcw), :] = resized_cropped[:]

    out_dir = join(output_base_path, class_id, modelname,
                   output_image_dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file_path = join(out_dir, '{:03d}.jpg'.format(cnt))
    dummy_camera_out_file_path = join(out_dir, 'cameras.npz')
    dummy_camera_mat = np.ones([3, 3]).astype(np.float64)
    dummy_world_mat = np.ones([3, 3]).astype(np.float64)
    np.savez(dummy_camera_out_file_path,
             camera_mat_0=dummy_camera_mat,
             world_mat_0=dummy_world_mat)
    skimage.io.imsave(out_file_path, new_image)
    pbar.update(1)
    """
    u = float(yidx.min()) / float(h)
    b = float(h - yidx.max()) / float(h)
    r = float(xidx.min()) / float(w)
    l = float(w - xidx.max()) / float(w)
    """

# %%
