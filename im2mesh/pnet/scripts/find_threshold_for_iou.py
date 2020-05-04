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
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import threading

from datetime import datetime

date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(OrderedDict, represent_odict)


def construct_odict(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_constructor('tag:yaml.org,2002:map', construct_odict)


#for thidx, th in enumerate(tqdm(thresholds)):
def testmap(inputs):
    trainer, model, th, dataset, test_loader, samples = inputs
    trainer.threshold = th
    # Handle each dataset separately
    eval_dicts = []
    print('th', th)
    for it, data in enumerate(test_loader):
        #for it, data in enumerate(tqdm(test_loader)):
        if it > samples:
            break
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
        eval_data = trainer.eval_step(data)
        eval_dict.update(eval_data)

    # Create pandas dataframe and save
    eval_df = pd.DataFrame(eval_dicts)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.groupby(by=['class name']).mean()

    # Print results
    iou = eval_df_class['iou'].mean().item()
    return {th: iou}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        help='Do not use cuda.')
    parser.add_argument('--dontcopy',
                        action='store_true',
                        help='Do not use cuda.')
    parser.add_argument('--no_copy_but_create_new',
                        action='store_true',
                        help='Do not use cuda.')
    parser.add_argument('--parallel',
                        type=int,
                        default=4,
                        help='Path to config file.')

    # Get configuration and basic arguments
    args, unknown_args = parser.parse_known_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')

    eval_utils.update_dict_with_options(cfg, unknown_args)
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    # Shorthands

    if args.dontcopy:
        out_dir = cfg['training']['out_dir']
    else:
        out_dir = os.path.join('out', cfg['data']['input_type'],
                               os.path.basename(args.config).split('.')[0])
        cfg['training']['out_dir'] = out_dir
        base_out_dir = cfg['training']['out_dir']
        out_dir = os.path.join(
            os.path.dirname(base_out_dir).replace('out',
                                                  'out/submission/eval'),
            os.path.basename(base_out_dir)) + '_' + datetime.now().strftime(
                ('%Y%m%d_%H%M%S'))
    print('out dir for eval: ', out_dir)
    if not args.dontcopy:
        if not os.path.exists(out_dir):
            if args.no_copy_but_create_new:
                os.makedirs(out_dir)
            else:
                #shutil.copytree(base_out_dir, out_dir)
                os.makedirs(out_dir)
                best_file = cfg['test']['model_file']
                best_path = os.path.join(base_out_dir, best_file)
                shutil.copy2(best_path, out_dir)
        else:
            raise ValueError('out dir already exists')
    if not args.dontcopy:
        patch_path = os.path.join(out_dir, 'diff.patch')
        subprocess.run('git diff > {}'.format(patch_path), shell=True)
        weight_path = os.path.join(out_dir, cfg['test']['model_file'])
        with open(weight_path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        cfg['test']['model_file_hash'] = md5
        yaml.dump(
            cfg,
            open(
                os.path.join(out_dir,
                             'find_th_config_{}.yaml'.format(date_str)), 'w'))

    out_file = os.path.join(out_dir, 'eval_full.pkl')
    out_file_class = os.path.join(out_dir, 'eval.csv')
    if args.dontcopy:
        t = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_file = out_file.replace('.pkl', '_' + t + '.pkl')
        out_file_class = out_file.replace('.csv', '_' + t + '.csv')
    # Dataset
    dataset = config.get_dataset('val', cfg, return_idx=True)
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

    print('Evaluating networks...')
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)
    thresholds = [
        *np.arange(0.3, 0.9, 0.05).tolist(), 0.99, 0.999, 0.9999, 0.99999,
        0.999999, 0.9999999, 0.99999999, 0.999999999
    ]
    samples = 1000000
    results = []
    max_iou = -1

    model.share_memory()
    values = [(trainer, model, th, dataset, test_loader, samples)
              for th in thresholds]

    with mp.get_context("spawn").Pool(args.parallel) as p:
        results = p.map(testmap, values)
    print(results)
    max_iou = -1
    best_th = 0
    for res in results:
        th, iou = next(iter(res.items()))

        if iou > max_iou:
            best_th = th
            max_iou = iou

    print('best iou={}, threshold={}'.format(max_iou, best_th))

    if not args.dontcopy:
        with open(os.path.join(out_dir, 'threshold'), 'w') as f:
            print(best_th, file=f)
            print(max_iou, file=f)
