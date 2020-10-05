# %%
import glob
import os
from tqdm import tqdm
import skimage.io
import skimage.transform
import shutil
import numpy as np

from joblib import Parallel, delayed
os.chdir('/home/mil/kawana/workspace/occupancy_networks')


# %%
def convert(inputs):
    indices, paths = inputs
    for idx in tqdm(indices):
        src_path, dst_path = paths[idx]
        image = skimage.io.imread(src_path)
        if image.ndim == 2:
            m = skimage.transform.resize(image, [128, 128])
        else:
            mr = skimage.transform.resize(image, [128, 128, image.shape[-1]])
            m = mr.mean(-1)
        mg = (m[None, ...] / 255).astype(np.float32)

        assert list(mg.shape) == [1, 128, 128
                                  ] and mg.dtype == np.float32, (mg.shape,
                                                                 mg.dtype)

        dirname = os.path.dirname(dst_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        np.savez(dst_path, image=mg)


# %%

bsp_img_dir = 'bsp_class_agnostic_margin_224'
img_dir = bsp_img_dir.replace('bsp_', '')
data_dir = 'data/Pix3DBSP'

src_paths = glob.glob('data/Pix3D/**/*.jpg', recursive=True)

indices = np.arange(len(src_paths))
split_indices_set = np.array_split(indices, 30)
dst_paths = [
    src_path.replace('Pix3D',
                     'Pix3DBSP').replace(img_dir,
                                         bsp_img_dir).replace('.jpg', '.npz')
    for src_path in src_paths
]

paths = [(src_path, dst_path)
         for src_path, dst_path in zip(src_paths, dst_paths)]
Parallel(n_jobs=30)(
    [delayed(convert)((split_indices_set[i], paths)) for i in range(30)])

# %%
text_src_paths = glob.glob('data/Pix3D/**/master.lst', recursive=True)

text_dst_paths = [
    text_src_path.replace('Pix3D', 'Pix3DBSP')
    for text_src_path in text_src_paths
]

for src_path, dst_path in zip(text_src_paths, text_dst_paths):
    shutil.copy(src_path, dst_path)
