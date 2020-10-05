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

dotenv.load_dotenv('/home/mil/kawana/workspace/occupancy_networks/.env',
                   verbose=True)
os.environ['WANDB_PROJECT'] = 'periodic_shape_occupancy_networks'

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--use_written_out_dir',
                    action='store_true',
                    help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--data_parallel',
                    action='store_true',
                    help='Train with data parallel.')
parser.add_argument(
    '--exit-after',
    type=int,
    default=-1,
    help='Checkpoint and exit after specified number of seconds'
    'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Set t0
t0 = time.time()

# Shorthands
#out_dir = cfg['training']['out_dir']
if args.use_written_out_dir:
    out_dir = cfg['training']['out_dir']
else:
    out_dir = os.path.join('out', cfg['data']['input_type'],
                           os.path.basename(args.config).split('.')[0])
cfg['training']['out_dir'] = out_dir
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)

if 'debug' in cfg['data']:
    train_shuffle = cfg['data']['debug'].get('train_shuffle', False)
else:
    train_shuffle = True

if args.data_parallel:
    dist_coef = 1
else:
    dist_coef = 1

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size * dist_coef,
                                           num_workers=4,
                                           shuffle=train_shuffle,
                                           collate_fn=data.collate_remove_none,
                                           worker_init_fn=data.worker_init_fn,
                                           drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=cfg['training']['val_batch_size'] * dist_coef,
    num_workers=4,
    shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=cfg['training'].get('vis_batch_size', 1) * dist_coef,
    shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
data_vis = next(iter(vis_loader))

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)
# Intialize training
npoints = 1000
learning_rate = float(cfg['training'].get('learning_rate', 1e-4)) * dist_coef
if not args.data_parallel:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)

if cfg['training'].get('skip_load_pretrained_optimizer', False):
    print('skip loading optimizer')
    checkpoint_io = CheckpointIO(out_dir, model=model)
elif not args.data_parallel:
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
else:
    assert args.data_parallel and not cfg['training'].get(
        'skip_load_pretrained_optimizer', False)
    checkpoint_io = CheckpointIO(out_dir, model=model)
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    CheckpointIO(out_dir, optimizer=optimizer)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get('loss_val_best',
                                -model_selection_sign * np.inf)

# Hack because of previous bug in code
# TODO: remove, because shouldn't be necessary
if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

# TODO: remove this switch
# metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f' %
      (model_selection_metric, metric_val_best))

# TODO: reintroduce or remove scheduler?
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000,
#                                       gamma=0.1, last_epoch=epoch_it)
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

run_id = cfg['training']['wandb_resume']
if run_id is not None:
    wandb.init(resume=run_id)
else:
    wandb.init()
wandb.config.update(cfg, allow_val_change=True)
# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
wandb.watch(model)
print('Total number of parameters: %d' % nparameters)

is_linear_decay = 'learning_rage_decay_at' in cfg['training']
current_lr = learning_rate
kill_epoch = cfg['training'].get('kill_epoch_at', np.inf)
while True:
    epoch_it += 1

    #     scheduler.step()
    if is_linear_decay and epoch_it in cfg['training'][
            'learning_rage_decay_at']:
        print('Decay learning rate from {} to {}'.format(
            current_lr, current_lr / 10))
        current_lr /= 10
        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
        checkpoint_io.module_dict['optimizer'] = optimizer
        trainer.optimizer = optimizer

    for batch in train_loader:
        it += 1
        losses = trainer.train_step(batch)
        for k, v in losses.items():
            logger.add_scalar('train/%s' % k, v, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            wandblog = {'train/{}'.format(k): v for (k, v) in losses.items()}
            wandb.log(wandblog, step=it)

            loss_str = ', '.join(
                ['{}={:.4f}'.format(k, v) for (k, v) in wandblog.items()])
            print('[Epoch %02d] it=%03d, %s' % (epoch_it, it, loss_str))

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0 and it > 0:
            print('Visualizing')
            trainer.visualize(data_vis, it=it, epoch_it=epoch_it)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0) and it > 0:
            print('Saving checkpoint')
            checkpoint_io.save('model.pt',
                               epoch_it=epoch_it,
                               it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0) and it > 0:
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it,
                               epoch_it=epoch_it,
                               it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0 and it > 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f' %
                  (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)
            wandblog = {'val/{}'.format(k): v for (k, v) in eval_dict.items()}
            wandb.log(wandblog, step=it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt',
                                   epoch_it=epoch_it,
                                   it=it,
                                   loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() -
                               t0) >= exit_after or kill_epoch < epoch_it:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt',
                               epoch_it=epoch_it,
                               it=it,
                               loss_val_best=metric_val_best)
            exit(3)
