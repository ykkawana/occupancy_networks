# %%
import matplotlib.pyplot as plt
import pandas
import pickle
import numpy as np
import yaml
from collections import defaultdict

from paper_resources import utils
# %%
models_config_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/model_configs.yaml'
fig_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization/resources/num_primitives_performance/primitive_performance.png'
model_names = ['AtlasNetV2', 'BSPNet', 'SHNet']
model_names_for_plot = {
    'AtlasNetV2': 'AtlasNetV2',
    'BSPNet': 'BSP-Net',
    'SHNet': 'NSDN'
}
primitive_nums = [10, 15, 20, 30, 50]

# %%
fscore_results = defaultdict(lambda: {})
semseg_results = defaultdict(lambda: {})
semseg_results_std = defaultdict(lambda: {})

configs = yaml.load(open(models_config_path, 'r'))
for model_name in model_names:
    for pnum in primitive_nums:
        config_dict = configs[model_name + '_' + str(pnum)]
        fscore = pickle.load(open(config_dict['fscore'], 'rb'))
        part_semseg = pickle.load(open(config_dict['part_semseg'], 'rb'))
        fscore_results[model_name][pnum] = fscore.groupby(
            'class name')['fscore_th=0.01 (mesh)'].mean().mean().item()
        #if model_name == 'SHNet' and pnum == 15:
        #    fscore_results[model_name][pnum] = 0.52
        semseg_results[model_name][pnum] = part_semseg.groupby(
            'class_name').mean().T.mean().mean().item()
        semseg_results_std[model_name][pnum] = part_semseg.groupby(
            'class_name').std().T.mean().mean().item()

# %%
fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
for model_name in model_names:
    values = [
        float(utils.cutdeci(fscore_results[model_name][num] * 100, deci=2))
        for num in primitive_nums
    ]
    ax.plot(primitive_nums, values, marker='o', ms=12)
ax.set_xlabel('# primitives', fontsize=35)
ax.set_ylabel('F-score', fontsize=35)
#ax.set_xticks(primitive_nums)
ax.set_xticks([10, 30, 50])
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
fig.legend(model_names_for_plot.values(),
           fontsize=30,
           ncol=3,
           columnspacing=0.2,
           bbox_to_anchor=(0.9, 1.1),
           loc='upper right',
           handletextpad=-0.2,
           borderaxespad=0.)
ax = fig.add_subplot(1, 2, 2)
for model_name in model_names:
    values = [
        float(utils.cutdeci(semseg_results[model_name][num] * 100, deci=2))
        for num in primitive_nums
    ]
    yerr = [
        float(
            utils.cutdeci(semseg_results_std[model_name][num] * 100 * 0.5,
                          deci=2)) for num in primitive_nums
    ]

    #ax.errorbar(primitive_nums, values, yerr=yerr, marker='o', ms=12, capsize=4, capthick=4)
    ax.plot(primitive_nums, values, marker='o', ms=12)
ax.set_xlabel('# primitives', fontsize=30)
ax.set_ylabel('Label IoU', fontsize=30)
ax.set_xticks([10, 30, 50])
#ax.set_xticks(primitive_nums)
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
#ax.set_xticklabels(primitive_nums)

fig.show()
dirname = os.path.dirname(fig_path)
if not os.path.exists(dirname):
    os.makedirs(dirname)
plt.savefig(fig_path, bbox_inches="tight")
# %%
