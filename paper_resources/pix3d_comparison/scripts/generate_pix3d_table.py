# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import pickle
import subprocess
import glob


def join(*args):
    return os.path.join(*args)


# %%
resource_base_dir_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/pix3d_comparison/resources'
fscore_values = [0.005, 0.0107337006427915, 0.05, 0.1, 0.2]
fscore_display_values = ['0.5\%', '1\%', '5\%', '10\%', '20\%']
sample_generation_classes = [
    'chair', 'table', 'bed', 'bookcase', 'misc', 'sofa'
]

shapenetv1_path = '/data/unagi0/kawana/workspace/ShapeNetCore.v1'
shapenetv2_path = '/data/unagi0/kawana/workspace/ShapeNetCore.v2'
shapenetocc_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
pix3d_base_path = '/home/mil/kawana/workspace/occupancy_networks/data/Pix3D'
rendering_script_path = '/home/mil/kawana/workspace/occupancy_networks/scripts/render_3dobj.sh'

side_length_scale = 0.0107337006427915
ours_name = 'PSNet30'
theirs_name = 'OccNet'

rendering_out_base_dir = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/pix3d_comparison'

pix3d_df_path = join(pix3d_base_path, 'pix3d_*.pkl')
rendering_out_dir = os.path.join(rendering_out_base_dir, 'resources')
rendering_gt_mesh_cache_dir = os.path.join(rendering_out_base_dir, 'cache')

id_to_dir_path_map = {
    'OccNet':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained',
    'PSNet30':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200417_123951'
}
id_to_mesh_dir_name = {
    'OccNet':
    'pretrained_pix3d_class_agnostic_margin_224',
    'PSNet30':
    'generation_pix3d_class_agnostic_margin_224_explicit_20200417_163408'
}
id_to_fscore_map = {
    'OccNet': 'eval_fscore_from_meshes.csv',
    'PSNet30': 'eval_fscore_from_meshes_explicit.csv'
}
id_to_full_fscore_map = {
    'OccNet': 'eval_fscore_from_meshes_full.pkl',
    'PSNet30': 'eval_fscore_from_meshes_full_explicit.pkl'
}
id_to_cd1_map = {
    'OccNet': 'eval_meshes_full.csv',
    'PSNet30': 'eval_meshes_full_explicit.csv'
}
ids = list(id_to_cd1_map.keys())
fscore_key_strs = ['fscore_th={} (mesh)'.format(val) for val in fscore_values]

# %%
data = []
for idx in ids:
    fdf = pd.read_csv(
        os.path.join(id_to_dir_path_map[idx], id_to_mesh_dir_name[idx],
                     id_to_fscore_map[idx]))
    data.append(
        OrderedDict({
            disp_val: fdf[key].mean()
            for disp_val, key in zip(fscore_display_values, fscore_key_strs)
        }))
df = pd.DataFrame(data, columns=fscore_display_values)

# %%
body = ""
names = ' & '.join(['Threshold(%)', *df.columns]) + " \\ \hline"
body += names
for idx in range(len(df)):
    method_id = ids[idx]

    def cutdeci(s):
        if isinstance(s, str):
            return s
        return "{:.3f}".format(s)

    els = [method_id, *map(cutdeci, df.loc[idx].tolist())]
    row = ' & '.join(els) + " \\ "
    body += ('\n' + row)

print(body)

with open(os.path.join(resource_base_dir_path, 'table.txt'), 'w') as f:
    print(body.replace('\\', '\\\\').replace('\\\h',
                                             '\h').replace('\\\mu', '\mu'),
          file=f)

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

# %%
oursdf = pickle.load(
    open(
        os.path.join(id_to_dir_path_map[ours_name],
                     id_to_mesh_dir_name[ours_name],
                     id_to_full_fscore_map[ours_name]), 'rb'))

theirsdf = pickle.load(
    open(
        os.path.join(id_to_dir_path_map[theirs_name],
                     id_to_mesh_dir_name[theirs_name],
                     id_to_full_fscore_map[theirs_name]), 'rb'))

# %%
pix3d_df_paths = glob.glob(pix3d_df_path)
dfs = []
for path in pix3d_df_paths:
    pix3d_df = pickle.load(open(path, 'rb'))
    dfs.append(pix3d_df)
pix3ddf = pd.concat(dfs)

for class_name in sample_generation_classes:
    class_id = label_to_synset[class_name]

    oursdf_cls = oursdf[oursdf['class id'] == class_id]
    theirsdf_cls = theirsdf[theirsdf['class id'] == class_id]

    assert len(oursdf_cls) == len(theirsdf_cls), (len(oursdf_cls),
                                                  len(theirsdf_cls))

    fscore_key = 'fscore_th={} (mesh)'.format(0.005)
    """
    oursdf_cls['diff'] = (oursdf_cls[fscore_key] - theirsdf_cls[fscore_key])

    filter = oursdf_cls['diff'] > 0
    idx = oursdf_cls[filter]['diff'].argmax()
    model_id = oursdf_cls[filter]['modelname'].iloc[idx]
    """

    model_idx = oursdf_cls[fscore_key].argmax()
    print(model_idx, class_name)
    print(oursdf_cls['modelname'].iloc[model_idx])
    model_id = oursdf_cls['modelname'].iloc[model_idx]

    #filter = oursdf_cls[fscore_key] > 0.5
    #idx = oursdf_cls[filter]['diff'].argmax()
    #model_id = oursdf_cls[filter]['modelname'].iloc[idx]

    model_paths = {
        idx: os.path.join(id_to_dir_path_map[idx], id_to_mesh_dir_name[idx],
                          'meshes', label_to_synset[class_name],
                          str(model_id) + '.off')
        for idx in ids
    }

    model_paths['gt'] = os.path.join(pix3d_base_path, class_id, model_id,
                                     'model.off')

    if not os.path.exists(rendering_out_dir):
        os.makedirs(rendering_out_dir)

    camera_param_path = os.path.join(rendering_out_base_dir,
                                     'camera_param.txt')
    for idx in model_paths:
        command = 'sh {script} {camera_param} {model} {out_dir} {idx}'.format(
            script=rendering_script_path,
            camera_param=camera_param_path,
            model=model_paths[idx],
            out_dir=rendering_out_dir,
            idx=idx + '_' + class_name)
        print(command)
        subprocess.run(command, shell=True)

# %%
