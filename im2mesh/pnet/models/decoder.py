import torch.nn as nn
import torch.nn.functional as F
import torch
from im2mesh.layers import (ResnetBlockFC, CResnetBlockConv1d, CBatchNorm1d,
                            CBatchNorm1d_legacy, ResnetBlockConv1d)
import sys
sys.path.insert(0, './external.periodic_shapes')
from periodic_shapes.models import super_shape, super_shape_sampler, periodic_shape_sampler, sphere_sampler
import time


class PeriodicShapeDecoderSimplest(nn.Module):
    ''' Decoder with CBN class 2.

    It differs from the previous one in that the number of blocks can be
    chosen.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
    '''
    def __init__(
        self,
        dim=3,
        z_dim=0,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        max_m=4,
        n_primitives=6,
        is_train_periodic_shape_sampler=False,
        shape_sampler_decoder_factor=1,
        is_quadrics=False,
        is_sphere=False,
        is_shape_sampler_sphere=False,
        is_feature_angles=True,
        is_feature_radius=True,
        is_feature_coord=True,
        transition_range=3.,
        paramnet_class='ParamNet',
        paramnet_hidden_size=128,
        paramnet_dense=True,
        is_single_paramnet=False,
        layer_depth=4,
        skip_position=3,  # count start from input fc
        is_skip=True,
        shape_sampler_decoder_class='PrimitiveWiseGroupConvDecoder',
        disable_learn_pose_but_transition=False,
        freeze_primitive=False,
        no_last_bias=False,
        supershape_freeze_rotation_scale=False,
        get_features_from=[],
        concat_input_feature_with_pose_feature=False):
        super().__init__()
        assert dim in [2, 3]
        self.is_train_periodic_shape_sampler = is_train_periodic_shape_sampler
        self.get_features_from = get_features_from
        self.concat_input_feature_with_pose_feature = concat_input_feature_with_pose_feature

        self.primitive = super_shape.SuperShapes(
            max_m,
            n_primitives,
            latent_dim=c_dim,
            train_logits=(not is_quadrics),
            train_ab=False,
            quadrics=is_quadrics,
            sphere=is_sphere,
            transition_range=transition_range,
            paramnet_class=paramnet_class,
            paramnet_hidden_size=paramnet_hidden_size,
            paramnet_dense=paramnet_dense,
            is_single_paramnet=is_single_paramnet,
            layer_depth=layer_depth,
            is_skip=is_skip,
            skip_position=skip_position,
            dim=dim,
            supershape_freeze_rotation_scale=supershape_freeze_rotation_scale,
            get_features_from=get_features_from)
        if freeze_primitive:
            self.primitive.require_grad = False

        if get_features_from:
            if concat_input_feature_with_pose_feature:
                feature_dim = paramnet_hidden_size + c_dim
            else:
                feature_dim = paramnet_hidden_size
        else:
            feature_dim = c_dim
        self.p_sampler = periodic_shape_sampler.PeriodicShapeSampler(
            feature_dim,
            max_m,
            n_primitives,
            dim=dim,
            factor=shape_sampler_decoder_factor,
            no_encoder=True,
            disable_learn_pose_but_transition=disable_learn_pose_but_transition,
            is_shape_sampler_sphere=is_shape_sampler_sphere,
            decoder_class=shape_sampler_decoder_class,
            is_feature_angles=is_feature_angles,
            is_feature_coord=is_feature_coord,
            is_feature_radius=is_feature_radius,
            no_last_bias=no_last_bias)
        # simple_sampler = super_shape_sampler.SuperShapeSampler(max_m,
        #                                                        n_primitives,
        #                                                        dim=dim)
        self.simple_sampler = sphere_sampler.SphereSampler(max_m,
                                                           n_primitives,
                                                           dim=dim)

    def forward(self, coord, _, color_feature, angles=None, **kwargs):
        params = self.primitive(color_feature)

        if self.is_train_periodic_shape_sampler:
            if self.get_features_from:
                if len(self.get_features_from) == 1:
                    feature = params[self.get_features_from[0] + '_feature']

                    if self.concat_input_feature_with_pose_feature:
                        feature = torch.cat([feature, color_feature], axis=-1)
                else:
                    raise NotImplementedError
            else:
                feature = color_feature

            assert feature.shape[-1] == 256 + 128
            output = self.p_sampler(params,
                                    thetas=angles,
                                    coord=coord,
                                    points=feature,
                                    return_surface_mask=True)
        else:
            output = self.simple_sampler(params,
                                         thetas=angles,
                                         coord=coord,
                                         points=color_feature,
                                         return_surface_mask=True)
        #shapes = [[10, 12, 900, 3], [10, 12, 900], [10, 12, 2048]]  #,
        #[10, 12, 10800]]
        """
        output = [
            torch.ones(shape, device=color_feature.device).float() *
            color_feature.mean() for shape in shapes
        ]
        output.append(None)
        """
        return output

    def to(self, device):
        super().to(device)
        self.p_sampler.to(device)
        self.simple_sampler.to(device)
        self.primitive.to(device)
        return self
