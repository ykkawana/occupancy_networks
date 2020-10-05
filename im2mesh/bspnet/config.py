import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.bspnet import models, training, generation
from im2mesh import data
from im2mesh import config
from bspnet import modelSVR


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']

    assert decoder == 'PretrainedBSPNetDecoder'

    if encoder == 'idx':
        raise NotImplementedError
    elif encoder is not None:
        assert encoder == 'PretrainedBSPNetEncoder'
    else:
        encoder = None

    model = modelSVR.bsp_network(32, 4096, decoder_kwargs['n_primitives'], 64,
                                 256)
    model.to(device)
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
                               **cfg.get('trainer', {}))

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
        is_gen_primitive_wise_watertight_mesh=cfg['generation'].get(
            'is_gen_primitive_wise_watertight_mesh', False),
        is_gen_primitive_wise_watertight_mesh_debugged=cfg['generation'].get(
            'is_gen_primitive_wise_watertight_mesh_debugged', False),
        is_gen_whole_mesh=cfg['generation'].get('is_gen_whole_mesh', False),
        is_gen_skip_vertex_attributes=cfg['generation'].get(
            'is_gen_skip_vertex_attributes', False),
        is_gen_watertight_mesh=cfg['generation'].get('is_gen_integrated_mesh',
                                                     False),
        is_gen_implicit_mesh=cfg['generation'].get('is_gen_implicit_mesh',
                                                   False),
        is_gen_surface_points=cfg['generation'].get('is_gen_surface_points',
                                                    False),
        is_just_measuring_time=cfg['generation'].get('is_just_measuring_time',
                                                     False),
        is_skip_realign=cfg['generation'].get('bspnet', {
            'is_skip_realign': False
        }).get('is_skip_realign', False),
        is_fit_to_gt_loc_scale=cfg['generation'].get('is_fit_to_gt_loc_scale',
                                                     False),
        preprocessor=preprocessor)
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
    pointcloud_transform = data.SubsamplePointcloud(
        cfg['data']['pointcloud_target_n'])
    if cfg.get('sdf_generation', False):
        pointcloud_transform = None

    fields['pointcloud'] = data.PointCloudField(
        cfg['data']['pointcloud_file'],
        pointcloud_transform,
        cfg,
        with_transforms=True,
        force_disable_bspnet_mode=cfg['data']['bspnet'].get(
            'force_disable_bspnet_mode', False))
    if cfg['test'].get('is_eval_semseg', False):
        fields['labeled_pointcloud'] = data.PartLabeledPointCloudField(
            cfg['data']['semseg_pointcloud_file'], cfg)

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
