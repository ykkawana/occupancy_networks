# %%
import os
import glob

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
occ_root = 'data/ShapeNet'

occ_list = []
for class_id in synset_to_label:
    occ_list.extend([
        (class_id, modelname)
        for modelname in os.listdir(os.path.join(occ_root, class_id))
        if os.path.isdir(os.path.join(occ_root, class_id, modelname))
    ])

occ_set = set(occ_list)
# %%
atv2_list = []
for class_id in synset_to_label:
    atv2_list.extend([
        (class_id, os.path.basename(path).replace('.points.ply', ''))
        for path in glob.glob(
            os.path.join(atv2_root, class_id, 'ply/*.points.ply'))
    ])

atv2_set = set(atv2_list)

# %%
print(len(occ_set), len(atv2_set))

# %%
for class_id in synset_to_label:
    train_list = []
    test_list = []
    for txt in ['train.lst', 'val.lst', 'test.lst']:
        dirpath = os.path.join(occ_root, class_id, txt)
        with open(dirpath) as f:
            modelnames = [(class_id, modelname.strip())
                          for modelname in f.readlines()]

        intersection = set(modelnames).intersection(atv2_set)
        if txt in ['train.lst', 'val.lst']:
            train_list.extend(list(intersection))
        else:
            test_list.extend(list(intersection))

    with open(os.path.join(atv2_root, class_id, 'train.lst'), 'w') as f:
        train_list = [modelname for class_id, modelname in train_list]
        f.write('\n'.join(train_list))
    with open(os.path.join(atv2_root, class_id, 'test.lst'), 'w') as f:
        test_list = [modelname for class_id, modelname in test_list]
        f.write('\n'.join(test_list))
