import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.atlasnetv2 import models, training, generation
from im2mesh import data
from im2mesh import config
from atlasnetv2.auxiliary.utils import weights_init


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_pointcloud = cfg['model'].get('encoder_pointcloud', None)
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']

    decoder = models.decoder_dict[decoder](dim=dim,
                                           z_dim=z_dim,
                                           c_dim=c_dim,
                                           **decoder_kwargs)

    if z_dim != 0:
        encoder_latent = models.encoder_latent_dict[encoder_latent](
            dim=dim, z_dim=z_dim, c_dim=c_dim, **encoder_latent_kwargs)
    else:
        encoder_latent = None

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](c_dim=c_dim, **encoder_kwargs)
        if encoder_pointcloud is not None:
            encoder_pointcloud = encoder_dict[encoder_pointcloud](
                npoint=cfg['data']['pointcloud_target_n'],
                nlatent=c_dim,
                **encoder_kwargs)
    else:
        encoder = None

    p0_z = get_prior_z(cfg, device)
    model = models.AtlasNetV2(decoder,
                              encoder,
                              encoder_latent,
                              encoder_pointcloud,
                              p0_z,
                              device=device)
    if cfg['data'][
            'input_type'] == 'pointcloud' or encoder_pointcloud is not None and cfg[
                'training'].get('atlasnetv2', {}).get('training_step',
                                                      None) == 'autoencoder':
        model.apply(weights_init)
    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(model,
                               optimizer,
                               device=device,
                               input_type=input_type,
                               vis_dir=vis_dir,
                               threshold=threshold,
                               eval_sample=cfg['training']['eval_sample'],
                               debugged=cfg['training'].get('debugged', False),
                               training_step=cfg['training'].get(
                                   'atlasnetv2',
                                   {}).get('training_step', None),
                               **cfg['trainer'])

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        preprocessor=preprocessor,
        debugged=cfg['training'].get('debugged', False),
        is_fit_to_gt_loc_scale=cfg['generation'].get('is_fit_to_gt_loc_scale',
                                                     False),
        training_step=cfg['training'].get('atlasnetv2',
                                          {}).get('training_step', None),
        point_scale=cfg['trainer']['point_scale'])
    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(torch.zeros(z_dim, device=device),
                       torch.ones(z_dim, device=device))

    return p0_z


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['model']['use_camera']

    fields = {}
    fields['points'] = data.PointsField(
        cfg['data']['points_file'],
        points_transform,
        with_transforms=with_transforms,
        unpackbits=cfg['data']['points_unpackbits'],
    )

    fields['patch'] = data.PlanarPatchField(
        mode, cfg['data'].get('patch_side_length', 20),
        cfg['data'].get('is_generate_mesh', False))

    if cfg['test'].get('is_eval_semseg', False):
        fields['labeled_pointcloud'] = data.PartLabeledPointCloudField(
            cfg['data']['semseg_pointcloud_file'], cfg)

    pointcloud_transform = data.SubsamplePointcloud(
        cfg['data']['pointcloud_target_n'])
    if cfg.get('sdf_generation', False):
        pointcloud_transform = None

    fields['pointcloud'] = data.PointCloudField(cfg['data']['pointcloud_file'],
                                                pointcloud_transform,
                                                cfg=cfg,
                                                with_transforms=True)

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
