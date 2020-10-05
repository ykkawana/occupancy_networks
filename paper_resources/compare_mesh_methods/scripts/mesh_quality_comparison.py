# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import trimesh
import os
import pandas
import pickle
import subprocess
import kaolin as kal
import kaolin.conversions.meshconversions as mesh_cvt
from kaolin.transforms import pointcloudfunc as pcfunc
from kaolin.transforms import transforms as tfs
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys
sys.path.insert(0, '/home/mil/kawana/workspace/occupancy_networks')
from im2mesh.utils import binvox_rw
import math
import numpy as np
from paper_resources import utils
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

shapenetv1_path = '/data/unagi0/kawana/workspace/ShapeNetCore.v1'
shapenetv2_path = '/data/unagi0/kawana/workspace/ShapeNetCore.v2'
shapenetocc_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
side_length_scale = 0.01

# %%
class_name = 'airplane'
base_eval_dir = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200413_015954'
rendering_script_path = '/home/mil/kawana/workspace/occupancy_networks/scripts/render_3dobj.sh'
rendering_out_base_dir = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/compare_mesh_methods'
rendering_out_dir = os.path.join(rendering_out_base_dir, 'resources')
rendering_gt_mesh_cache_dir = os.path.join(rendering_out_base_dir, 'cache')
mesh_dir_60k = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_f60k_20200511_163104'

id_to_mesh_dir_names = {
    'd0':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_0_20200509_155825',
    'd1':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_1_20200509_155936',
    'd2':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_2_20200509_160000',
    'd3':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_3_20200509_160021',
    'd4':
    #'/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812',
    #'/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153',
    'u0':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up0_20200505_052830',
    'u1':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up1_20200505_052940',
    'u2':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up2_20200505_053138',
    'b':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_primitive_wise_watertight_20200507_023808'
}
id_to_fscore_pkl_map = {
    'd0':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_0_20200509_155825/eval_fscore_from_meshes_full_mesh_iou_normalize_sample100k__explicit_20200509_203111.pkl',
    'd1':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_1_20200509_155936/eval_fscore_from_meshes_full_mesh_iou_normalize_sample100k__explicit_20200509_203322.pkl',
    'd2':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_2_20200509_160000/eval_fscore_from_meshes_full_mesh_iou_normalize_sample100k__explicit_20200509_203350.pkl',
    'd3':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_3_20200509_160021/eval_fscore_from_meshes_full_mesh_iou_normalize_sample100k__explicit_20200509_203504.pkl',
    'd4':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/eval_fscore_from_meshes_full_normalize__explicit_20200505_231208.pkl',
    'u0':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up0_20200505_052830/eval_fscore_from_meshes_full_mesh_iou_normalize__20200509_161520.pkl',
    'u1':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up1_20200505_052940/eval_fscore_from_meshes_full_mesh_iou_normalize__20200509_161255.pkl',
    'u2':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up2_20200505_053138/eval_fscore_from_meshes_full_mesh_iou_normalize__20200506_052902.pkl',
    'b':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_primitive_wise_watertight_20200507_023808/eval_fscore_from_meshes_full_mesh_iou_normalize__20200507_031956.pkl'
}
del id_to_fscore_pkl_map['d1']
del id_to_fscore_pkl_map['d3']
id_to_time_pkl_map = {
    'd0':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_0_20200509_155825/eval_gen_time_full_20200509_093822.pkl',
    'd1':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_1_20200509_155936/eval_gen_time_full_20200509_093923.pkl',
    'd2':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_2_20200509_160000/eval_gen_time_full_20200509_094025.pkl',
    'd3':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_3_20200509_160021/eval_gen_time_full_20200509_094128.pkl',
    'd4':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/eval_gen_time_full_20200509_094231.pkl',
    'u0':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up0_20200505_052830/eval_gen_time_full_20200509_084209.pkl',
    'u1':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up1_20200505_052940/eval_gen_time_full_20200509_084332.pkl',
    'u2':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up2_20200505_053138/eval_gen_time_full_20200509_084955.pkl',
    'b':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_primitive_wise_watertight_20200507_023808/eval_gen_time_full_20200509_094338.pkl'
}
dmarker = 'o'
umarker = 'x'
bmarker = '*'
markers_map = {
    'd0': dmarker,
    'd1': dmarker,
    'd2': dmarker,
    'd3': dmarker,
    'd4': dmarker,
    'u0': umarker,
    'u1': umarker,
    'u2': umarker,
    'b': bmarker
}

legend_texts_map = {
    'd0': 'NSDN ico0',
    'd1': 'NSDN ico1',
    'd2': 'NSDN ico2',
    'd3': 'NSDN ico3',
    'd4': 'NSDN ico4',
    'u0': 'MISE up0',
    'u1': 'MISE up1',
    'u2': 'MISE up2',
    'b': 'BSPNet'
}

ids = list(id_to_fscore_pkl_map.keys())

best_mesh_dir_id = 'd4'
worst_mesh_dir_id = 'u2'

gdf = pickle.load(
    open(
        os.path.join(base_eval_dir, id_to_mesh_dir_names[best_mesh_dir_id],
                     id_to_fscore_pkl_map[best_mesh_dir_id]), 'rb'))
gdf = gdf[gdf['class id'] == label_to_synset[class_name]]

wdf = pickle.load(
    open(
        os.path.join(base_eval_dir, id_to_mesh_dir_names[worst_mesh_dir_id],
                     id_to_fscore_pkl_map[worst_mesh_dir_id]), 'rb'))
wdf = wdf[wdf['class id'] == label_to_synset[class_name]]

assert len(gdf) == len(wdf), (len(gdf), len(wdf))

fscore_key = 'fscore_th={} (mesh)'.format(side_length_scale)

gdf['diff'] = (gdf[fscore_key] - wdf[fscore_key])

filter = gdf[fscore_key] > 0.9
idx = gdf[filter][fscore_key].argmax()
#idx = gdf[filter]['diff'].argmax()
model_id = gdf[filter]['modelname'].iloc[idx]


# %%
def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


# %%
model_paths = {
    idx: os.path.join(base_eval_dir, id_to_mesh_dir_names[idx], 'meshes',
                      label_to_synset[class_name],
                      str(model_id) + '.off')
    for idx in ids
}
model_path_60k = os.path.join(mesh_dir_60k, 'meshes',
                              label_to_synset[class_name],
                              str(model_id) + '.off')
shapenet_model_path = os.path.join(shapenetv1_path,
                                   label_to_synset[class_name], model_id,
                                   'model.obj')
#shapenet_model_path = os.path.join(shapenetv2_path, label_to_synset[class_name], model_id, 'models', 'model_normalized.solid.binvox')
#shapenet_model_path = os.path.join(shapenetocc_path, label_to_synset[class_name], model_id, 'model.binvox')
gt_mesh_cache_path = os.path.join(rendering_gt_mesh_cache_dir,
                                  model_id + '.obj')
if not os.path.exists(gt_mesh_cache_path) or True:
    mesh = as_mesh(trimesh.load(shapenet_model_path))
    min3 = [0] * 3
    max3 = [0] * 3

    for i in range(3):
        min3[i] = np.min(mesh.vertices[:, i])
        max3[i] = np.max(mesh.vertices[:, i])

    bb_min, bb_max = tuple(min3), tuple(max3)

    # Get extents of model.
    bb_min, bb_max = np.array(bb_min), np.array(bb_max)
    total_size = (bb_max - bb_min).max()

    # Set the center (although this should usually be the origin already).
    centers = np.array([[(bb_min[0] + bb_max[0]) / 2,
                         (bb_min[1] + bb_max[1]) / 2,
                         (bb_min[2] + bb_max[2]) / 2]])
    # Scales all dimensions equally.
    scale = total_size

    mesh.vertices -= centers
    mesh.vertices *= 1. / scale
    mesh.vertices *= (1 + side_length_scale)
    mesh.export(gt_mesh_cache_path)
model_paths['gt'] = gt_mesh_cache_path

f_scores = {}
for idx in ids:
    df = pickle.load(
        open(
            os.path.join(base_eval_dir, id_to_mesh_dir_names[idx],
                         id_to_fscore_pkl_map[idx]), 'rb'))
    f_score = df.groupby('class id').mean().mean()[
        'fscore_th={} (mesh)'.format(0.01)].item()
    f_scores[idx] = f_score

times = {}
for idx in ids:
    df = pickle.load(open(id_to_time_pkl_map[idx], 'rb'))
    #times[idx] = df[df['class id'] ==
    #                label_to_synset[class_name]]['mesh'].mean().item()
    times[idx] = df.groupby('class id').mean().mean().iloc[1:].sum()

# %%
times_plot = [times[idx] for idx in ids]
f_scores_plot = [f_scores[idx] for idx in ids]
markers = [markers_map[idx] for idx in ids]
legend_texts = [legend_texts_map[idx] for idx in ids]

fig = plt.figure(figsize=(6.4, 3))
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
for t, f, m, idx in zip(times_plot, f_scores_plot, markers, ids):
    ax.scatter(t, f, marker=m, s=100, label=idx)
    #ax.scatter(math.log(t), f, marker=m, s=100, label=idx)
ax.set_xscale('log')
ax.legend(legend_texts,
          fontsize=15,
          ncol=4,
          columnspacing=0.2,
          bbox_to_anchor=(1.05, 1.35),
          loc='upper right',
          handletextpad=-0.2,
          borderaxespad=0.)
ax.set_xlabel('time (sec)', fontsize=20)
ax.set_ylabel('F-score', fontsize=20)
#ax.set_xticklabels([t for t in times_plot])
#plt.locator_params(axis='x', nbins=5)
ax.set_xticks([0.01, 0.1, 1, 5])

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

fig.savefig(os.path.join(rendering_out_dir, 'plot.png'),
            bbox_inches="tight",
            pad_inches=0.05)

# %%
if not os.path.exists(rendering_out_dir):
    os.makedirs(rendering_out_dir)

camera_param_path = os.path.join(rendering_out_base_dir, 'camera_param.txt')
for idx in model_paths:
    command = 'sh {script} {camera_param} {model} {out_dir} {idx}'.format(
        script=rendering_script_path,
        camera_param=camera_param_path,
        model=model_paths[idx],
        out_dir=rendering_out_dir,
        idx=idx)
    print(command)
    subprocess.run(command, shell=True)

command = 'sh {script} {camera_param} {model} {out_dir} {idx}'.format(
    script=rendering_script_path,
    camera_param=camera_param_path,
    model=model_path_60k,
    out_dir=rendering_out_dir,
    idx='60k')
print(command)
subprocess.run(command, shell=True)
# %%
id_to_verts_faces_pkl_map = {
    idx: os.path.join(os.path.dirname(id_to_fscore_pkl_map[idx]),
                      'surface_verts_faces_count.pkl')
    for idx in ids
}

table_text = '&$\#$V&$\#$F&F-score&time\\ \hline \n'
for idx in ids:
    pkl_path = id_to_verts_faces_pkl_map[idx]
    vfdf = pickle.load(open(pkl_path, 'rb')).mean()
    val = f_scores[idx]
    line = '{model_name} & {verts} & {faces} & {fscore} & {time} \\ \n'
    text = line.format(
        model_name=legend_texts_map[idx],
        #verts=int(vfdf['verts_num'].item()),
        #faces=int(vfdf['faces_num'].item()),
        verts=(int(vfdf['verts_out'].item()) //
               100 if int(vfdf['verts_out'].item()) >= 100 else utils.cutdeci(
                   vfdf['verts_out'].item() / 100, deci=1)),
        faces=(int(vfdf['faces_out'].item()) //
               100 if int(vfdf['faces_out'].item()) >= 100 else utils.cutdeci(
                   vfdf['faces_out'].item() / 100, deci=1)),
        fscore=utils.cutdeci(f_scores[idx] * 100, deci=2),
        time=utils.cutdeci(times[idx], deci=3))
    if idx == 'b':
        els = text.split('&')
        gray_els = ['\textcolor[gray]{0.5}{' + el + '}' for el in els]
        text = '&'.join(gray_els)
    table_text += text
with open(os.path.join(rendering_out_dir, 'table.txt'), 'w') as f:
    print(table_text.replace('\\', '\\\\').replace('\c', 'c').replace(
        '\t', 't').replace('\h', 'h'),
          file=f)
