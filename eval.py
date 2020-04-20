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


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(OrderedDict, represent_odict)


def construct_odict(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_constructor('tag:yaml.org,2002:map', construct_odict)

parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

# Get configuration and basic arguments
args, unknown_args = parser.parse_known_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
for idx, arg in enumerate(unknown_args):
    if arg.startswith('--'):
        arg = arg.replace('--', '')
        value = unknown_args[idx + 1]
        keys = arg.split('.')
        if keys[0] not in cfg:
            cfg[keys[0]] = {}
        child_cfg = cfg.get(keys[0], {})
        for key in keys[1:]:
            item = child_cfg.get(key, None)
            if isinstance(item, dict):
                child_cfg = item
            elif item is None:
                child_cfg[key] = value
            else:
                child_cfg[key] = type(item)(value)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
if '--dontcopy' in unknown_args:
    out_dir = cfg['training']['out_dir']
else:
    base_out_dir = cfg['training']['out_dir']
    out_dir = os.path.join(
        os.path.dirname(base_out_dir).replace('out', 'out/submission/eval'),
        os.path.basename(base_out_dir)) + '_' + datetime.now().strftime(
            ('%Y%m%d_%H%M%S'))
print('out dir for eval: ', out_dir)
if not '--dontcopy' in unknown_args:
    if not os.path.exists(out_dir):
        shutil.copytree(base_out_dir, out_dir)
    else:
        raise ValueError('out dir already exists')

if not '--dontcopy' in unknown_args:
    patch_path = os.path.join(out_dir, 'diff.patch')
    subprocess.run('git diff > {}'.format(patch_path), shell=True)
    weight_path = os.path.join(out_dir, cfg['test']['model_file'])
    with open(weight_path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    cfg['test']['model_file_hash'] = md5
    yaml.dump(cfg, open(os.path.join(out_dir, 'config.yaml'), 'w'))

out_file = os.path.join(out_dir, 'eval_full.pkl')
out_file_class = os.path.join(out_dir, 'eval.csv')

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
print(model)
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

    inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
    angles = data.get('angles').to(device)
    points = data.get('points').to(device)

    feature = model.encode_inputs(inputs)

    kwargs = {}
    scaled_coord = points * cfg['trainer']['pnet_point_scale']
    output = model.decode(scaled_coord, None, feature, angles=angles, **kwargs)
    super_shape_point, surface_mask, sgn, sgn_BxNxNP, radius = output

    eval_dicts.append(eval_dict)
    eval_data = trainer.eval_step(data)
    eval_dict.update(eval_data)

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
