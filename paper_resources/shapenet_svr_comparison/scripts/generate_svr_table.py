# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import os

# %%
header = """\begin{table}
    \centering
    \resizebox{\textwidth}{!}{\begin{tabular}{c|c|ccccccccccccc|c}
    \bhline{1.5 pt}
"""

footer = """
\bhline{1.5 pt}

    \end{tabular}}
    \caption{Results. Significantly outperforms existing methods in chamfer distance. Comparable in IoU.}
\end{table}
"""

# %%
resource_base_dir_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/shapenet_svr_comparison'
csv_dir_path = os.path.join(resource_base_dir_path, 'csv')
onet_new_fscore_table_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained/pretrained/eval_fscore_from_meshes.csv'
fscore_key = 'fscore_th=0.01 (mesh)'
id_to_dir_path_map = {
    #'Sphere30':
    #'/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_oceff10_dense_normal_loss_pointcloud_n_4096_20200414_051415',
    'SHNet':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_f60k_20200511_163104'
}

id_to_fscore_map = {
    #'Sphere30': 'eval_fscore_from_meshes_explicit.csv',
    'SHNet':
    'eval_fscore_from_meshes_mesh_iou_normalize__explicit_20200511_214918.csv'
}
id_to_cd1_iou_map = {
    #'Sphere30': 'eval_meshes_explicit.csv',
    'SHNet': 'eval_meshes_mesh_iou_normalize__explicit_20200512_003806.csv'
}
ids = list(id_to_cd1_iou_map.keys())

# %%
ioudf = pd.read_csv(os.path.join(csv_dir_path, 'iou.csv'))
cd1df = pd.read_csv(os.path.join(csv_dir_path, 'cd1.csv'))
fdf = pd.read_csv(os.path.join(csv_dir_path, 'fscore.csv'))

#onet_new_fscore_df = pd.read_csv(onet_new_fscore_table_path)[fscore_key] * 100
#new_data = {key: value for key, value in zip(fdf.columns, ['OccNet*', *onet_new_fscore_df.tolist(), onet_new_fscore_df.mean().item()])}
#fdf = fdf.append(new_data, ignore_index=True)

for idx in ids:
    s = pd.read_csv(
        os.path.join(id_to_dir_path_map[idx],
                     id_to_fscore_map[idx]))[fscore_key]
    s = s * 100
    new_data = {
        key: value
        for key, value in zip(
            fdf.columns, [idx, *s.tolist(), s.mean().item()])
    }
    fdf = fdf.append(new_data, ignore_index=True)

for idx in ids:
    s = pd.read_csv(
        os.path.join(id_to_dir_path_map[idx],
                     id_to_cd1_iou_map[idx]))['chamfer-L1 (mesh)']
    new_data = {
        key: value
        for key, value in
        zip(cd1df.columns,
            [idx, *s.tolist(), s.mean().item()])
    }
    cd1df = cd1df.append(new_data, ignore_index=True)

for idx in ids:
    s = pd.read_csv(
        os.path.join(id_to_dir_path_map[idx],
                     id_to_cd1_iou_map[idx]))['iou (mesh)']
    new_data = {
        key: value
        for key, value in
        zip(ioudf.columns,
            [idx, *s.tolist(), s.mean().item()])
    }
    ioudf = ioudf.append(new_data, ignore_index=True)

# %%


def cutdeci(s, deci=3):
    if isinstance(s, str):
        return s
    deci_str = "{" + ":.{}".format(deci) + "f}"
    return deci_str.format(s)


def bold_if_neccessary(els, df, dfidx, elsidx, is_max=True, deci=3):
    try:
        num = float(els)
    except:
        return els
    strs = [cutdeci(s, deci=deci) for s in df.iloc[:, elsidx].tolist()]
    floats = map(float, strs)
    is_bold = (is_max and num == max(floats)) or (not is_max
                                                  and num == min(floats))
    if is_bold:
        return '{\bf ' + els + '}'
    else:
        return els


# %%
body = ""
names = ' & '.join(['', '', *cd1df.columns[1:]]) + " \\ \hline"
body += names
for idx in range(len(cd1df)):
    if idx == 0:
        r1 = "\multirow{" + str(len(cd1df)) + "}{*}{CD1}"
    else:
        r1 = ""

    els = map(cutdeci, cd1df.loc[idx].tolist())
    row = r1
    for elsidx, els in enumerate(els):
        row += (' & ' +
                bold_if_neccessary(els, cd1df, idx, elsidx, is_max=False))
    row += " \\ "
    body += ('\n' + row)
body += '\hline'

for idx in range(len(fdf)):
    if idx == 0:
        r1 = "\multirow{" + str(len(fdf)) + "}{*}{F-score}"
    else:
        r1 = ""

    els = [cutdeci(s, deci=2) for s in fdf.loc[idx].tolist()]
    row = r1
    for elsidx, els in enumerate(els):
        row += (' & ' +
                bold_if_neccessary(els, fdf, idx, elsidx, is_max=True, deci=2))
    row += " \\ "

    body += ('\n' + row)
body = body + '\hline'

for idx in range(len(ioudf)):
    if idx == 0:
        r1 = "\multirow{" + str(len(ioudf)) + "}{*}{IoU}"
    else:
        r1 = ""

    els = map(cutdeci, ioudf.loc[idx].tolist())
    row = r1
    for elsidx, els in enumerate(els):
        row += (' & ' +
                bold_if_neccessary(els, ioudf, idx, elsidx, is_max=True))

    row += " \\ "
    body += '\n' + row

with open(os.path.join(resource_base_dir_path, 'table.txt'), 'w') as f:
    print(body.replace('\\', '\\\\').replace('\\\h', '\h').replace(
        '\b', '\\b').replace('\\\mu', '\mu'),
          file=f)

# %%
