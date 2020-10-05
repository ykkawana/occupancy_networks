import torch
from torch import nn
from periodic_shapes.models import super_shape_sampler, sphere_sampler
from periodic_shapes import utils
from periodic_shapes.layers import super_shape_functions
from periodic_shapes.models import point_net
from periodic_shapes.models import decoder

EPS = 1e-7


#class PeriodicShapeSampler(super_shape_sampler.SuperShapeSampler):
class PeriodicShapeSampler(sphere_sampler.SphereSampler):
    def __init__(self,
                 num_points,
                 *args,
                 last_scale=.1,
                 factor=1,
                 act='leaky',
                 decoder_class='PrimitiveWiseGroupConvDecoder',
                 is_shape_sampler_sphere=False,
                 spherical_angles=False,
                 no_encoder=False,
                 is_feature_angles=True,
                 is_feature_coord=True,
                 is_feature_radius=True,
                 no_last_bias=False,
                 return_sdf=False,
                 is_simpler_sgn=False,
                 is_infer_r1r2=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.clamp = True
        self.factor = factor
        self.num_points = num_points
        self.num_labels = (1 if not is_infer_r1r2 or self.dim == 2 else 2
                           )  # Only infer r2 for 3D
        self.theta_dim = 2 if self.dim == 2 else 4
        self.last_scale = last_scale
        self.is_simpler_sgn = is_simpler_sgn
        self.act = act
        self.no_encoder = no_encoder
        self.is_shape_sampler_sphere = is_shape_sampler_sphere
        self.spherical_angles = spherical_angles
        self.return_sdf = return_sdf
        self.is_infer_r1r2 = is_infer_r1r2
        if self.is_infer_r1r2:
            assert not self.is_shape_sampler_sphere and not self.spherical_angles

        c64 = 64 // self.factor
        self.encoder_dim = c64 * 2
        if not self.no_encoder:
            self.encoder = point_net.PointNet(self.encoder_dim,
                                              dim=self.dim,
                                              factor=self.factor,
                                              act=self.act)
            shaper_decoder_input_dim = self.encoder_dim + self.theta_dim + 1
            decoder_class = 'PrimitiveWiseGroupConvDecoderWithNoEncoder',
        #elif decoder_class == 'PrimitiveWiseGroupConvDecoder':
        #    shaper_decoder_input_dim = num_points + self.theta_dim + 1 + (
        #        self.dim if is_feature_coord else 0)
        else:
            shaper_decoder_input_dim = self.num_points
        self.decoder = decoder.decoder_dict[decoder_class](
            self.n_primitives,
            shaper_decoder_input_dim,
            self.num_labels,
            dim=self.dim,
            factor=self.factor,
            act=self.act,
            is_feature_coord=is_feature_coord,
            is_feature_radius=is_feature_radius,
            is_feature_angles=is_feature_angles,
            no_last_bias=no_last_bias)

    def transform_circumference_angle_to_super_shape_world_cartesian_coord(
        self, thetas, radius, primitive_params, *args, points=None, **kwargs):
        assert not points is None

        assert len(thetas.shape) == 3, thetas.shape
        B, P, D = thetas.shape
        thetas_reshaped = thetas.view(B, 1, P, D)

        assert len(radius.shape) == 4, radius.shape
        # r = (B, n_primitives, P, dim - 1)
        r = radius.view(B, self.n_primitives, P, D)

        if self.is_shape_sampler_sphere and self.spherical_angles:
            coord = super_shape_functions.sphere2cartesian(r, thetas_reshaped)
        else:
            coord = super_shape_functions.polar2cartesian(r, thetas_reshaped)
        assert not torch.isnan(coord).any()
        # posed_coord = B, n_primitives, P, dim
        if self.learn_pose:
            posed_coord = self.project_primitive_to_world(
                coord, primitive_params)
        else:
            posed_coord = coord

        #print('')
        #print('get pr from points')
        periodic_net_r = self.get_periodic_net_r(thetas.unsqueeze(1), points,
                                                 r[..., -1], posed_coord)

        final_r = r.clone()
        if self.is_shape_sampler_sphere and self.spherical_angles:
            #print('mean r1 in points', r[..., 0].mean())
            final_r[..., 0] = r[..., 0] + periodic_net_r.squeeze(-1)
            #print('mean final r in points', final_r[..., 0].mean())

        elif self.is_infer_r1r2:
            final_r = r + periodic_net_r
        else:
            #print('mean r1 in points', r[..., -1].mean())
            final_r[..., -1] = r[..., -1] + periodic_net_r.squeeze(-1)
            #print('mean final r in points', final_r[..., -1].mean())

        if self.clamp:
            final_r = final_r.clamp(min=EPS)
        else:
            final_r = torch.relu(final_r) + EPS
        #print('r in ss stats',
        #      final_r.mean().item(),
        #      final_r.max().item(),
        #      final_r.min().item(),
        #      final_r.std().item())

        # B, n_primitives, P, dim
        if self.is_shape_sampler_sphere and self.spherical_angles:
            cartesian_coord = super_shape_functions.sphere2cartesian(
                final_r, thetas_reshaped)
        else:
            cartesian_coord = super_shape_functions.polar2cartesian(
                final_r, thetas_reshaped)

        assert [*cartesian_coord.shape] == [B, self.n_primitives, P, self.dim]
        assert not torch.isnan(cartesian_coord).any()

        if self.learn_pose:
            posed_cartesian_coord = self.project_primitive_to_world(
                cartesian_coord, primitive_params)
        else:
            posed_cartesian_coord = cartesian_coord
        # B, n_primitives, P, dim
        return posed_cartesian_coord

    def get_periodic_net_r(self, thetas, points, radius, coord):
        # B, 1 or N, P, dim - 1
        #print('mean coord in pr', coord.mean())
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert points.shape[0] == thetas.shape[0]

        B = points.shape[0]
        assert points.shape[1] == self.num_points

        _, _, P, _ = thetas.shape
        # B, P, N

        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        assert [*coord.shape] == [B, self.n_primitives, P, self.dim]

        if self.no_encoder:
            encoded = points
        else:
            encoded = self.encoder(points).view(B, 1, 1,
                                                self.encoder_dim).repeat(
                                                    1, P, self.n_primitives, 1)

        radius = self.decoder(encoded, thetas, radius, coord)
        radius = radius * self.last_scale

        #print('mean from pr ', radius.mean())
        return radius

    def get_indicator(self,
                      x,
                      y,
                      z,
                      r1,
                      r2,
                      theta,
                      phi,
                      params,
                      *args,
                      points=None,
                      **kwargs):
        assert not points is None
        coord_list = [x, y]
        is3d = len(z.shape) == len(x.shape)
        if is3d:
            # 3D case
            coord_list.append(z)
            angles = torch.stack([theta, phi], axis=-1)
            radius = r2
        else:
            angles = theta.unsqueeze(-1)
            radius = r1
        coord = torch.stack(coord_list, axis=-1)

        if self.is_shape_sampler_sphere and self.spherical_angles:
            posed_coord = super_shape_functions.sphere2cartesian(
                torch.stack([r1, r2], axis=-1), angles)
        else:
            posed_coord = super_shape_functions.polar2cartesian(
                torch.stack([r1, r2], axis=-1), angles)
        # posed_coord = B, n_primitives, P, dim
        if self.learn_pose:
            posed_coord = self.project_primitive_to_world(posed_coord, params)

        #print('get pr from sgn')
        rp = self.get_periodic_net_r(angles, points, radius, posed_coord)

        #print('mean r1 in sgn', r1.mean())
        numerator = (coord**2).sum(-1)
        if self.is_shape_sampler_sphere:
            r1 = r1 + rp.squeeze(-1)
            #print('mean final r in sgn', r1.mean())
            if self.clamp:
                r1 = r1.clamp(min=EPS)
                nep = numerator.clamp(min=EPS)
            else:
                r1 = nn.functional.relu(r1) + EPS
                nep = (numerator + EPS)
            if self.return_sdf:
                dist = nep.sqrt() - r1
                indicator = dist.sign() * dist**2
            else:
                indicator = 1 - numerator.clamp(min=EPS).sqrt() / r1

        else:
            if is3d:
                if self.is_infer_r1r2:
                    r1 = r1 + rp[..., 0]
                    r2 = r2 + rp[..., -1]
                    if self.clamp:
                        r1 = r1.clamp(min=EPS)
                        r2 = r2.clamp(min=EPS)
                    else:
                        r1 = nn.functional.relu(r1) + EPS
                        r2 = nn.functional.relu(r2) + EPS
                else:
                    r2 = r2 + rp.squeeze(-1)
                    if self.clamp:
                        r2 = r2.clamp(min=EPS)
                    else:
                        r2 = nn.functional.relu(r2) + EPS
            else:
                r1 = r1 + rp.squeeze(-1)
                if self.clamp:
                    r1 = r1.clamp(min=EPS)
                else:
                    r1 = nn.functional.relu(r1) + EPS
            if self.is_simpler_sgn:
                denominator = r2
            else:
                if self.clamp:
                    denominator = ((r1**2) * (r2**2) * (phi.cos()**2) +
                                   (r2**2) * (phi.sin()**2)).clamp(min=EPS)
                else:
                    denominator = ((r1**2) * (r2**2) * (phi.cos()**2) +
                                   (r2**2) * (phi.sin()**2)) + EPS
            if self.return_sdf:
                if self.clamp:
                    nep = numerator.clamp(min=EPS)
                else:
                    nep = (numerator + EPS)
                dist = nep.sqrt() - denominator.sqrt()
                indicator = (dist).sign() * dist**2
                denominator = r2
            elif self.is_simpler_sgn:
                if self.clamp:
                    indicator = 1. - numerator.clamp(
                        min=EPS).sqrt() / denominator

                else:
                    indicator = 1. - (numerator + EPS).sqrt() / denominator
            else:
                if self.clamp:
                    indicator = 1. - (numerator /
                                      denominator).clamp(min=EPS).sqrt()
                else:
                    indicator = 1. - (numerator / denominator + EPS).sqrt()

        return indicator

    def get_sgn(self, coord, params, *args, **kwargs):
        if self.is_shape_sampler_sphere and self.spherical_angles:
            r, angles = self.cartesian2sphere(coord, params, *args, **kwargs)
            r1 = r[..., 0]
            r2 = r[..., 1]
            theta = angles[..., 0]
            phi = angles[..., 1]
        else:
            r1, r2, theta, phi = self.cartesian2polar(coord, params, *args,
                                                      **kwargs)

        dim = coord.shape[-1]
        x = coord[..., 0]
        y = coord[..., 1]
        z = torch.zeros([1], device=coord.device) if dim == 2 else coord[...,
                                                                         2]
        indicator = self.get_indicator(x, y, z, r1, r2, theta, phi, params,
                                       *args, **kwargs)
        assert not torch.isnan(indicator).any(), indicator
        return indicator
