import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib
matplotlib.use('Agg')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
import wandb
import dotenv
from pykeops.torch import LazyTensor
from tqdm import tqdm

dotenv.load_dotenv(verbose=True)
sdf_dirpath = os.getenv('SDF_DIRPATH')


# Arguments
def train(gpu_idx, args, distributed=True):
    rank = gpu_idx

    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:{}".format(gpu_idx) if is_cuda else "cpu")

    # Set t0
    t0 = time.time()
    # Dataset
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg)
    test_dataset = config.get_dataset('test', cfg)

    whole_data = len(train_dataset) + len(val_dataset) + len(test_dataset)

    batch_size = cfg['training']['batch_size']

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)

    #t = train_dataset[0]
    pbar = tqdm(total=whole_data)
    for loader in [train_loader, val_loader, test_loader]:

        for batch in loader:
            surface_points = batch['pointcloud'].to(device)
            points = batch['points'].clone()
            sdf_points = batch['points'].to(device)
            sgn = -(batch['points.occ'].to(device) - 0.5) * 2
            #G_i1 = LazyTensor(primitive_points_BxPsNx1x2)
            #X_j1 = LazyTensor(target_points_Bx1xPtx2)
            G_i1 = LazyTensor(sdf_points.unsqueeze(2))
            X_j1 = LazyTensor(surface_points.unsqueeze(1))
            dist = G_i1.sqdist(X_j1).min(2).squeeze(-1) * sgn

            dist_cpu = dist.to('cpu')

            breakpoint()
            assert False

            for idx in range(dist_cpu.shape[0]):
                save_cache(dist_cpu[idx], points[idx],
                           batch['inputs.category'][idx],
                           batch['inputs.object'][idx])
                pbar.update(1)
    pbar.close()


def save_cache(dist, points, cat_id, ob_id):
    sdf_dirfullpath = os.path.join(sdf_dirpath, cat_id, ob_id)
    if not os.path.exists(sdf_dirfullpath):
        os.makedirs(sdf_dirfullpath)
    filepath = os.path.join(sdf_dirfullpath, 'sdf_points')
    dist = dist.numpy()
    points = points.numpy()
    np.savez(filepath, points=points, distances=dist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a 3D reconstruction model.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        help='Do not use cuda.')
    parser.add_argument(
        '--exit-after',
        type=int,
        default=-1,
        help='Checkpoint and exit after specified number of seconds'
        'with exit code 2.')
    parser.add_argument('--dist_port', type=int, default=8887)

    args = parser.parse_args()
    train(0, args, distributed=False)
