# %%
import glob
import os
import pickle

# %%
test_dir = 'disposable'
output_base_path = '/data/ugui0/kawana/ShapeNetLikePix3D_correct_direction'
output_image_dir_name = 'class_agnostic_margin_224'

# %%
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
all_hashes = []
for synset in synset_to_label:
    path = os.path.join(output_base_path, synset)
    model_paths = [
        path for path in glob.glob(os.path.join(path, '*'))
        if os.path.isdir(path)
    ]
    hashes = []
    for model_path in model_paths:
        modelname = os.path.basename(model_path)
        image_paths = glob.glob(
            os.path.join(model_path, output_image_dir_name, '*.jpg'))
        if len(image_paths) != 0 and os.path.exists(
                os.path.join(model_path, 'points.npz')) and os.path.exists(
                    os.path.join(
                        model_path, 'sdf_points.npz')) and os.path.exists(
                            os.path.join(model_path, 'pointcloud.npz')):
            hashes.append(modelname)
    with open(os.path.join(path, 'master.lst'), 'w') as f:
        for idx, line in enumerate(hashes):
            if idx == (len(hashes) - 1):
                print(line, file=f, end='')
            else:
                print(line, file=f)
    all_hashes.extend(hashes)

with open(os.path.join(output_base_path, 'master.lst'), 'w') as f:
    for line in all_hashes:
        print(line, file=f)
