# %%
import matplotlib.pyplot as plt
import pandas
import pickle
import numpy as np
import yaml
from collections import defaultdict, OrderedDict
import os
from paper_resources import utils
# %%
models_config_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/model_configs.yaml'
out_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/effect_of_loss/resources/table.txt'
our_attributes = OrderedDict({
    'SHNet_10_occupancy_loss': {
        'text': 'SHNet${}_O$',
        'imex': [True, False]
    },
    'SHNet_10_chamfer_loss': {
        'text': 'SHNet${}_C$',
        'imex': [False, True]
    },
    'SHNet_10_surface_loss': {
        'text': 'SHNet${}_S$',
        'imex': [True, True]
    }
})
their_attributes = {
    'AtlasNetV2_30': {
        'text': 'AtlasNetV2',
        'imex': [False, True]
    },
    'BSPNet_256': {
        'text': 'BSPNet',
        'imex': [True, False]
    }
}

# %%
fscore_results = {}

configs = yaml.load(open(models_config_path, 'r'))
table_text = '&implicit&explicit&F-score \\ \hline \n'
for idx, attributes in enumerate([our_attributes, their_attributes]):
    for model_name in attributes:
        config_dict = configs[model_name]
        fscore = pickle.load(open(config_dict['fscore'], 'rb'))
        val = fscore.groupby(
            'class name')['fscore_th=0.01 (mesh)'].mean().mean().item()
        line = '{model_name} & {im_check} & {ex_check} & {fscore} \\ \n'
        attrs = attributes[model_name]
        text = line.format(model_name=attrs['text'],
                           im_check='\checkmark' if attrs['imex'][0] else '',
                           ex_check='\checkmark' if attrs['imex'][1] else '',
                           fscore=utils.cutdeci(val * 100, deci=2))
        table_text += text
    if idx == 0:
        table_text += '\hline \n'
# %%
with open(out_path, 'w') as f:
    print(table_text.replace('\\', '\\\\').replace('\c',
                                                   'c').replace('\h', 'h'),
          file=f)
