# %%
import os
import pandas as pd
import numpy as np
# %%
pn30 = {
    'cd1':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153/eval_meshes_full_no_surface__explicit_20200503_180748.pkl',
    'cd1_surface':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153/eval_meshes_full___explicit_20200503_180748.pkl',
    'f':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153/eval_fscore_from_meshes_full_no_surface__explicit_20200503_180747.pkl',
    'f_surface':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153/eval_fscore_from_meshes_full___explicit_20200503_180748.pkl',
    'cd1_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153/eval_meshes_full_no_surface_normalize__explicit_20200503_225809.pkl',
    'cd1_surface_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153/eval_meshes_full_normalize__explicit_20200503_225621.pkl',
    'f_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153/eval_fscore_from_meshes_full_no_surface_normalize__explicit_20200503_225809.pkl',
    'f_surface_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153/eval_fscore_from_meshes_full_normalize__explicit_20200503_225707.pkl'
}
pn30_nono = {
    'cd1':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/eval_meshes_full_no_surface__explicit_20200503_180215.pkl',
    'cd1_surface':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/eval_meshes_full___explicit_20200503_180215.pkl',
    'cd1_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/eval_meshes_full_no_surface_normalize__explicit_20200503_225905.pkl',
    'cd1_surface_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/eval_meshes_full_normalize__explicit_20200503_225702.pkl',
    'f':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/eval_fscore_from_meshes_full_no_surface__explicit_20200503_180215.pkl',
    'f_surface':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/eval_fscore_from_meshes_full___explicit_20200503_180215.pkl',
    'f_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/eval_fscore_from_meshes_full_no_surface_normalize__explicit_20200503_225905.pkl',
    'f_surface_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/eval_fscore_from_meshes_full_normalize__explicit_20200503_225804.pkl'
}
occ = {
    'cd1':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403/eval_meshes_full_no_surface__20200503_170034.pkl',
    'cd1_surface':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403/eval_meshes_full.pkl',
    'cd1_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403/eval_meshes_full_no_surface_normalize__20200503_225750.pkl',
    'cd1_surface_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403/eval_meshes_full_no_surface_normalize__20200503_225750.pkl',
    'f':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403/eval_fscore_from_meshes_full_no_surface__20200503_165833.pkl',
    'f_surface':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403/eval_fscore_from_meshes_full.pkl',
    'f_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403/eval_fscore_from_meshes_full_no_surface_normalize__20200503_225750.pkl',
    'f_surface_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403/eval_fscore_from_meshes_full_normalize__20200503_225649.pkl'
}
bsp256 = {
    'cd1':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_20200501_185114/eval_meshes_full_no_surface__20200503_204324.pkl',
    'cd1_surface':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_20200501_185114/eval_meshes_full___20200503_204324.pkl',
    'cd1_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_20200501_185114/eval_meshes_full_no_surface_normalize__20200503_225855.pkl',
    'cd1_surface_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_20200501_185114/eval_meshes_full_normalize__20200503_225654.pkl',
    'f':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_20200501_185114/eval_fscore_from_meshes_full_no_surface__20200503_204324.pkl',
    'f_surface':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_20200501_185114/eval_fscore_from_meshes_full___20200503_204324.pkl',
    'f_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_20200501_185114/eval_fscore_from_meshes_full_no_surface_normalize__20200503_225855.pkl',
    'f_surface_normalize':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_20200501_185114/eval_fscore_from_meshes_full_normalize__20200503_225756.pkl'
}
# %%
cd1 = ['cd1', 'cd1_surface', 'cd1_normalize', 'cd1_surface_normalize']
f = ['f', 'f_surface', 'f_normalize', 'f_surface_normalize']
# %%
runs = {'pn30': pn30, 'pn30_nono': pn30_nono, 'occ': occ, 'bsp256': bsp256}
for met in f:
    for name, run in runs.items():
        df = pickle.load(open(run[met], 'rb'))
        #print(name, met, df.columns)
        try:
            label = [l for l in df.columns
                     if l.startswith('fscore_th=0.01')][0]
        except:
            continue
        m = df.groupby('class name').mean().mean()[label]
        print(name, met, m)
# %%
for met in cd1:
    for name, run in runs.items():
        df = pickle.load(open(run[met], 'rb'))
        #label = [l for l in df.columns if l.startswith('0.01')][0]
        m = df.groupby('class name').mean().mean()['chamfer-L1 (mesh)']
        print(name, met, m)

# %%
