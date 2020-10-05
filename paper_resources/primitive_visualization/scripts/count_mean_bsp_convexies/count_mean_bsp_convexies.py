import argparse
import os
import subprocess
import hashlib
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
import shutil
import yaml
from collections import OrderedDict
import eval_utils
from datetime import datetime
from bspnet import modelSVR

date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(OrderedDict, represent_odict)


def construct_odict(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_constructor('tag:yaml.org,2002:map', construct_odict)

parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--dontcopy', action='store_true', help='Do not use cuda.')
parser.add_argument('--no_copy_but_create_new',
                    action='store_true',
                    help='Do not use cuda.')
parser.add_argument('--use_config_in_eval_dir',
                    action='store_true',
                    help='Do not use cuda.')

# Get configuration and basic arguments
args, unknown_args = parser.parse_known_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

eval_utils.update_dict_with_options(cfg, unknown_args)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands

out_dir = os.path.dirname(args.config)

out_file = os.path.join(out_dir, 'convex_count.pkl')

if (args.dontcopy or args.use_config_in_eval_dir):
    out_file = out_file.replace('.pkl', '_' + date_str + '.pkl')
    out_file_class = out_file.replace('.csv', '_' + date_str + '.csv')
# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
try:
    checkpoint_io.load(cfg['test']['model_file'])
except FileExistsError:
    print('Model file does not exist. Exiting.')
    exit()

# Trainer
trainer = config.get_trainer(model, None, cfg, device=device)

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)

# Evaluate
model.eval()

eval_dicts = []
print('Evaluating networks...')
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          collate_fn=data.collate_remove_none,
                                          worker_init_fn=data.worker_init_fn)

gen_helper = modelSVR.BSPNetMeshGenerator(model, device=device)
# Handle each dataset separately
for it, data in enumerate(tqdm(test_loader)):
    if data is None:
        print('Invalid data.')
        continue
    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}

    modelname = model_dict['model']
    category_id = model_dict['category']

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    eval_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }

    eval_dicts.append(eval_dict)

    inputs = data.get('inputs', torch.empty(1, 0)).to(device)
    with torch.no_grad():
        out_m, t = gen_helper.encode(inputs, measure_time=True)
        model_float, t = gen_helper.eval_points(out_m, measure_time=True)
        num = gen_helper.get_number_of_primitives(model_float,
                                                  out_m,
                                                  measure_time=True)

    eval_dict.update({'convex_num': num})

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)
