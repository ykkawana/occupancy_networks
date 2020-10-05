# %%
import pandas as pd
import numpy as np
import os
import glob
from IPython.display import Image, display
from collections import defaultdict
import shutil
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
# %%
pkl_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/rebuttal/resources/hole_count.pkl'
data_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
df = pd.read_pickle(pkl_path)

# %%
class_name_indices = {
    'bench': {
        'indices': [2, 7, 22, 31],
        'holes': 5
    },
    'chair': {
        'indices': [*list(range(10000))],
        #'indices': [2, 13, 0, 36, 3, 8, 23],
        'holes': 1
        #'holes': 3
    }
}
class_id_modelnames = defaultdict(lambda: [])
res_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/rebuttal/resources/images'
if not os.path.exists(res_path):
    os.makedirs(res_path)
for class_name, info in class_name_indices.items():
    class_id = label_to_synset[class_name]
    df_holes = df[(df['holes'] > info['holes']) & (df['class_id'] == class_id)]

    print(class_id)
    for idx in info['indices']:
        print(idx)
        if len(df_holes) < idx - 1:
            continue
        modelname = df_holes.iloc[idx].modelname
        class_id_modelnames[class_id].append(modelname)
        image_path = os.path.join(data_path, class_id, modelname,
                                  'img_choy2016', '000.jpg')
        #display(Image(filename=image_path))
        shutil.copy(image_path, os.path.join(res_path, modelname + '.jpg'))
# %%
"""
for class_id, modelnames in class_id_modelnames.items():
    txt = '\n'.join(modelnames)
    path = os.path.join(data_path, class_id, 'holes_overfit.lst')
    print(path)

    with open(os.path.join(data_path, class_id, 'holes_overfit.lst'),
              'w') as f:
        f.write(txt)
"""
