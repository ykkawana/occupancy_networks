import glob
import os
import pandas as pd
from tqdm import tqdm
from skimage import io
import pickle
import numpy as np

data_dir_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
image_dir_name = 'img_choy2016'
out_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/pix3d_comparison'


def extract_only_dirs(files):
    f = [d for d in files if os.path.isdir(d)]
    return f


#whole_images_num = len(
#    glob.glob(os.path.join(data_dir_path, '**/*.jpg'), recursive=True))
whole_images_num = 1050792
df_list = [[]] * whole_images_num
class_paths = extract_only_dirs(glob.glob(os.path.join(data_dir_path, '*')))
pbar = tqdm(total=whole_images_num)
cnt = 0
for class_path in class_paths:
    obj_paths = extract_only_dirs(glob.glob(os.path.join(class_path, '*')))
    for obj_path in obj_paths:
        image_dir_names = os.path.join(data_dir_path, class_path, obj_path,
                                       'img_choy2016', '*.jpg')
        image_paths = glob.glob(image_dir_names)
        for image_path in image_paths:
            if cnt >= len(df_list):
                print('exit')
                break
            img = io.imread(image_path)
            class_id = os.path.basename(class_path)
            modelname = os.path.basename(obj_path)
            if img.ndim == 2:
                img = img[..., np.newaxis]
            img = img.mean(-1)
            if img.ndim != 2:
                print('ndim not 2')
                continue
            #img = img[:, 2:]
            h, w = img.shape
            yidx, xidx = np.where(img != 255)
            u = float(yidx.min()) / float(h)
            b = float(h - yidx.max()) / float(h)
            r = float(xidx.min()) / float(w)
            l = float(w - xidx.max()) / float(w)
            df_list[cnt] = [class_id, modelname, u, b, r, l, w, h]
            cnt += 1
            pbar.update(1)
df = pd.DataFrame(
    df_list, columns=['class_id', 'modelname', 'u', 'b', 'r', 'l', 'w', 'l'])
pickle.dump(df, open(os.path.join(out_path, 'margin.pkl'), 'wb'))
