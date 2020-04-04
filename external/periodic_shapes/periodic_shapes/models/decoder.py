import torch
from torch import nn
from periodic_shapes.layers import primitive_wise_layers
from im2mesh.layers import (ResnetBlockFC, CResnetBlockConv1d, CBatchNorm1d,
                            CBatchNorm1d_legacy, ResnetBlockConv1d,
                            ResnetBlockConv2d)

import torch.nn.functional as F
import numpy as np


class ShapeDecoderCBatchNorm(nn.Module):
    def __init__(self,
                 n_primitives,
                 feature_dim,
                 output_dim,
                 factor=1,
                 dim=3,
                 act='leaky',
                 is_feature_radius=True,
                 is_feature_coord=True,
                 is_feature_angles=True):
        super().__init__()
        self.n_primitives = n_primitives
        self.output_dim = output_dim
        self.act = act
        self.dim = dim
        self.feature_dim = feature_dim
        self.is_feature_angles = is_feature_angles
        self.is_feature_coord = is_feature_coord
        self.is_feature_radius = is_feature_radius

        c256 = 256 // factor
        legacy = False

        self.input_dim = (self.dim - 1) * 2 if self.is_feature_angles else 0
        self.input_dim += self.dim if self.is_feature_coord else 0
        self.input_dim += 1 if self.is_feature_radius else 0
        self.input_dim *= self.n_primitives

        self.fc_p = nn.Conv1d(self.input_dim, c256, 1)
        self.block0 = CResnetBlockConv1d(self.feature_dim, c256, legacy=legacy)
        self.block1 = CResnetBlockConv1d(self.feature_dim, c256, legacy=legacy)
        self.block2 = CResnetBlockConv1d(self.feature_dim, c256, legacy=legacy)
        self.block3 = CResnetBlockConv1d(self.feature_dim, c256, legacy=legacy)
        self.block4 = CResnetBlockConv1d(self.feature_dim, c256, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(self.feature_dim, c256)
        else:
            self.bn = CBatchNorm1d_legacy(self.feature_dim, c256)

        self.fc_out = nn.Conv1d(c256, self.output_dim * self.n_primitives, 1)

        if not act == 'leaky':
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, feature, thetas, radius, coord, *args, **kwargs):
        """
        Args:
            feature (B, feature_dim)
            thetas (B, 1 or N, P, dim - 1)
            radius (B, N, P)
            coord (B, N, P, dim)
        Returns:
            radius B, N, P, output_dim
        """
        # B, 1 or N, P, dim - 1
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert feature.shape[0] == thetas.shape[0]
        B, feature_dim = feature.shape
        _, N, P, Dn1 = thetas.shape
        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        assert feature.shape[1] == self.feature_dim
        assert coord.ndim == 4

        feature_list = []

        if self.is_feature_angles:
            transposed_thetas = thetas.transpose(2, 3).contiguous()
            sin = transposed_thetas.sin()
            cos = transposed_thetas.cos()
            # B, N x (dim - 1) * 2 = (2 or 4), P
            sincos = (torch.cat([sin, cos], axis=-1) * 100).expand(
                -1, -1, self.n_primitives,
                -1).view(B,
                         self.n_primitives * (self.dim - 1) * 2, P)

            feature_list.append(sincos)
        if self.is_feature_radius:
            transposed_radius = radius

            feature_list.append(transposed_radius)

        if self.is_feature_coord:
            transposed_target = coord.transpose(2, 3).contiguous().view(
                B, self.n_primitives * self.dim, P)
            feature_list.append(transposed_target)

        for f in feature_list:
            print(f.shape)
        x = torch.cat(feature_list, axis=1)

        net = self.fc_p(x)

        net = self.block0(net, feature)
        #net = self.block1(net, feature)
        #net = self.block2(net, feature)
        #net = self.block3(net, feature)
        net = self.block4(net, feature)

        out = self.fc_out(self.actvn(self.bn(net, feature)))

        radius = out.view(B, self.n_primitives, self.output_dim,
                          P).transpose(2, 3).contiguous()

        # B, n_primitives, P, self.label_num = 1
        print('radius', radius.mean().item())
        return radius


class PrimitiveWiseGroupConvDecoder(nn.Module):
    def __init__(self,
                 n_primitives,
                 input_dim,
                 output_dim,
                 factor=1,
                 dim=3,
                 act='leaky',
                 is_feature_radius=True,
                 is_feature_coord=True,
                 is_feature_angles=True,
                 no_last_bias=False):
        super().__init__()
        self.n_primitives = n_primitives
        self.output_dim = output_dim
        self.act = act
        self.dim = dim
        self.is_feature_angles = is_feature_angles
        self.is_feature_coord = is_feature_coord
        self.is_feature_radius = is_feature_radius
        self.is_feature_coord = is_feature_coord
        c64 = 64 // factor

        self.periodic_input_dim = (self.dim -
                                   1) * 2 if self.is_feature_angles else 0
        self.periodic_input_dim += self.dim if self.is_feature_coord else 0
        self.periodic_input_dim += 1 if self.is_feature_radius else 0

        self.input_dim = input_dim + self.periodic_input_dim

        self.decoder = nn.Sequential(
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      self.input_dim,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      self.output_dim,
                                                      act='none',
                                                      bias=(not no_last_bias)))

    def forward(self, feature, thetas, radius, coord, *args, **kwargs):

        #print('coord', coord.shape, coord.mean(), coord.max(), coord.min(),
        #      coord.std())
        #print('angles', thetas.sin().mean())
        #print('r', radius.mean())
        # B, 1 or N, P, dim - 1
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert feature.shape[0] == thetas.shape[0]
        B, N, P, Dn1 = thetas.shape
        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        _, feature_dim = feature.shape
        # B, 1 or N, dim - 1, P
        thetas_transposed = thetas.transpose(2, 3).contiguous()

        radius_transposed = radius.view(B, self.n_primitives, 1,
                                        P).contiguous()

        sin = thetas_transposed.sin()
        cos = thetas_transposed.cos()

        # B, N, dim - 1, P
        sincos = (torch.cat([sin, cos], axis=-2) * 100).expand(
            -1, self.n_primitives, -1, -1)

        # Feature dims has to be (B, P, self.n_primitives, feature dim)

        periodic_feature_list = [
            feature.view(B, 1, feature_dim, 1).expand(-1, self.n_primitives,
                                                      -1, P)
        ]
        if self.is_feature_angles:
            periodic_feature_list.append(sincos)

        if self.is_feature_radius:
            periodic_feature_list.append(radius_transposed)

        if self.is_feature_coord:
            transposed_target = coord.transpose(2, 3)  #.contiguous()
            periodic_feature_list.append(transposed_target)

        encoded_sincos = torch.cat(periodic_feature_list, axis=-2)
        radius = self.decoder(encoded_sincos).view(
            B, self.n_primitives, self.output_dim,
            P).transpose(2, 3).contiguous()

        # B, n_primitives, P, self.label_num = 1
        #return torch.ones_like(radius)
        #print('f r', radius.mean(), radius.median(), radius.min(),
        #      radius.max())
        return radius


class PrimitiveWiseGroupConvDecoderWithNoEncoder(nn.Module):
    def __init__(self,
                 n_primitives,
                 input_dim,
                 output_dim,
                 factor=1,
                 dim=3,
                 act='leaky',
                 **kwargs):
        super().__init__()
        self.n_primitives = n_primitives
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dim = dim
        c64 = 64 // factor
        self.decoder = nn.Sequential(
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      input_dim,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      self.output_dim,
                                                      act='none'))

    def forward(self, feature, thetas, radius, *args, **kwargs):
        # B, 1 or N, P, dim - 1
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert feature.shape[0] == thetas.shape[0]
        assert feature.ndim == 4
        B, _, _, feature_dim = feature.shape
        _, N, P, Dn1 = thetas.shape
        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        assert feature.shape[1] == P

        # B, P, N
        thetas_transposed = thetas.transpose(1, 2).contiguous()

        radius_transposed = radius.transpose(1, 2).contiguous().view(
            B, P, self.n_primitives, 1)

        # B, P, N, dim - 1
        sin = thetas_transposed.sin()
        cos = thetas_transposed.cos()

        # B, P, N, (dim - 1) * 2 = (2 or 4)
        sincos = (torch.cat([sin, cos], axis=-1) * 100).expand(
            -1, -1, self.n_primitives, -1)

        assert [*sincos.shape
                ] == [B, P, self.n_primitives,
                      (self.dim - 1) * 2], sincos.shape

        # Feature dims has to be (B, P, self.n_primitives, feature dim)
        feature_list = [feature, sincos, radius_transposed]
        encoded_sincos = torch.cat(feature_list, axis=-1)
        radius = self.decoder(encoded_sincos).view(B, P, self.n_primitives,
                                                   self.output_dim).transpose(
                                                       1, 2).contiguous()

        # B, n_primitives, P, self.label_num = 1
        return radius


class PrimitiveWiseGroupConvDecoderUseFCToExpandFeature(nn.Module):
    def __init__(self,
                 n_primitives,
                 input_dim,
                 output_dim,
                 factor=1,
                 dim=3,
                 act='leaky',
                 is_feature_radius=True,
                 is_feature_coord=True,
                 is_feature_angles=True):
        super().__init__()
        self.n_primitives = n_primitives
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dim = dim
        c64 = int(64 // factor)
        self.is_feature_angles = is_feature_angles
        self.is_feature_coord = is_feature_coord
        self.is_feature_radius = is_feature_radius
        self.decoder = nn.Sequential(
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      input_dim,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      self.output_dim,
                                                      act='none'))

        self.periodic_input_dim = (self.dim -
                                   1) * 2 if self.is_feature_angles else 0
        self.periodic_input_dim += self.dim if self.is_feature_coord else 0
        self.periodic_input_dim += 1 if self.is_feature_radius else 0
        self.fc = primitive_wise_layers.PrimitiveWiseLinear(
            self.n_primitives,
            self.periodic_input_dim,
            self.input_dim,
            act='none')

    def forward(self, feature, thetas, radius, coord, *args, **kwargs):
        # B, 1 or N, P, dim - 1
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert feature.shape[0] == thetas.shape[0]
        B, N, P, Dn1 = thetas.shape
        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        _, feature_dim = feature.shape
        assert feature_dim == self.input_dim
        # B, 1 or N, dim - 1, P
        thetas_transposed = thetas.transpose(2, 3)  #.contiguous()

        radius_transposed = radius.view(B, self.n_primitives, 1,
                                        P)  #.contiguous()

        sin = thetas_transposed.sin()
        cos = thetas_transposed.cos()

        # B, N, dim - 1, P
        sincos = (torch.cat([sin, cos], axis=-2) * 100).expand(
            -1, self.n_primitives, -1, -1)

        # Feature dims has to be (B, P, self.n_primitives, feature dim)
        periodic_feature_list = []
        if self.is_feature_angles:
            periodic_feature_list.append(sincos)

        if self.is_feature_radius:
            periodic_feature_list.append(radius_transposed)

        if self.is_feature_coord:
            transposed_target = coord.transpose(2, 3)  #.contiguous()
            periodic_feature_list.append(transposed_target)
        #for f in feature_list:
        #    print(f.shape)
        periodic_feature = torch.cat(periodic_feature_list,
                                     axis=-2).contiguous()

        # B, N, input_dim, P
        feature = feature.view(B, 1, self.input_dim,
                               1).contiguous() + self.fc(periodic_feature)
        radius = self.decoder(feature).view(B, self.n_primitives,
                                            self.output_dim,
                                            P).transpose(2, 3).contiguous()

        # B, n_primitives, P, self.label_num = 1
        return radius


class BatchNormDecoder(nn.Module):
    def __init__(self,
                 n_primitives,
                 input_dim,
                 output_dim,
                 factor=1,
                 dim=3,
                 act='leaky',
                 shape_sampler_layer_depth=3,
                 is_feature_radius=True,
                 is_feature_coord=True,
                 is_feature_angles=True):
        super().__init__()
        self.n_primitives = n_primitives
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dim = dim
        hidden_size = int(128 // factor)
        self.is_feature_angles = is_feature_angles
        self.is_feature_coord = is_feature_coord
        self.is_feature_radius = is_feature_radius
        self.shape_sampler_layer_depth = shape_sampler_layer_depth

        self.periodic_input_dim = (self.dim -
                                   1) * 2 if self.is_feature_angles else 0
        self.periodic_input_dim += self.dim if self.is_feature_coord else 0
        self.periodic_input_dim += 1 if self.is_feature_radius else 0
        """
        multires = 5
        embed_kwargs = {
            'include_input': True,
            'input_dims': self.dim,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.cos, torch.sin]
        }
        #freq_bands = 2.**torch.linspace(0., embed_kwargs['max_freq_log2'],
        #                                embed_kwargs['num_freqs'])
        # B, N, dim * P
        #self.freq_bands = freq_bands.view(1, 1, -1, 1).to('cuda')
        print('periodic_input_dim', self.periodic_input_dim)
        self.embedder = Embedder(**embed_kwargs)
        self.periodic_input_dim *= (multires * 2 + 1)
        print('periodic_input_dim', self.periodic_input_dim)
        """

        self.in_conv1d = nn.Conv1d(input_dim, hidden_size, 1)
        self.periodic_in_conv1d = nn.Conv1d(
            self.periodic_input_dim * self.n_primitives, hidden_size, 1)
        self.act = nn.LeakyReLU(0.2, True)
        self.out_conv1d = nn.Conv1d(hidden_size, n_primitives * output_dim, 1)
        self.bn = nn.BatchNorm1d(hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockConv1d(hidden_size)
            for _ in range(self.shape_sampler_layer_depth)
        ])

    def forward(self, feature, thetas, radius, coord, *args, **kwargs):
        # B, 1 or N, P, dim - 1
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert feature.shape[0] == thetas.shape[0]
        B, N, P, Dn1 = thetas.shape
        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        _, feature_dim = feature.shape
        assert feature_dim == self.input_dim
        # B, 1 or N, dim - 1, P
        thetas_transposed = thetas.transpose(2, 3)  #.contiguous()

        radius_transposed = radius.view(B, self.n_primitives, 1,
                                        P)  #.contiguous()

        sin = thetas_transposed.sin()
        cos = thetas_transposed.cos()

        # B, N, dim - 1, P
        sincos = (torch.cat([sin, cos], axis=-2) * 100).expand(
            -1, self.n_primitives, -1, -1)

        # Feature dims has to be (B, P, self.n_primitives, feature dim)
        periodic_feature_list = []
        if self.is_feature_angles:
            periodic_feature_list.append(sincos)

        if self.is_feature_radius:
            periodic_feature_list.append(radius_transposed)

        if self.is_feature_coord:
            transposed_target = coord.transpose(2, 3)  #.contiguous()
            periodic_feature_list.append(transposed_target)
        #for f in feature_list:
        #    print(f.shape)

        # B, N, feature_dim, P
        periodic_feature = torch.cat(periodic_feature_list,
                                     axis=-2).contiguous()

        #assert not torch.all(
        #    torch.eq(periodic_feature[:, 0, :, :], periodic_feature[:,
        #                                                            2, :, :]))

        #periodic_feature = self.embedder.embed(periodic_feature)
        """
        periodic_feature_raw = (
            self.freq_bands *
            periodic_feature.view(B, self.n_primitives, 1, -1) / 3)
        periodic_feature_raw = periodic_feature_raw.view(
            B, self.n_primitives, int(
                (self.periodic_input_dim - self.dim) / 2), P)
        periodic_feature_sin = torch.sin(periodic_feature_raw)
        periodic_feature_cos = torch.cos(periodic_feature_raw)
        periodic_feature = torch.cat(
            [periodic_feature_sin, periodic_feature_cos, periodic_feature],
            axis=-2)
        """

        # B, N, input_dim, P
        feature = self.in_conv1d(feature.view(B, self.input_dim, 1))
        periodic_feature = self.periodic_in_conv1d(
            periodic_feature.view(B,
                                  self.n_primitives * self.periodic_input_dim,
                                  P))

        net = feature + periodic_feature

        for block in self.blocks:
            net = block(net)

        radius = self.out_conv1d(self.act(self.bn(net))).view(
            B, self.n_primitives, self.output_dim,
            P).transpose(2, 3).contiguous()

        # B, n_primitives, P, self.label_num = 1
        return radius

    def to(self, device):
        super().to(device)
        #self.freq_bands = self.freq_bands.to(device)
        return self


class Embedder:
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        n_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, n_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        print('out_dim', self.out_dim)

    def embed(self, inputs):
        return torch.cat([fn(inputs * np.pi) for fn in self.embed_fns],
                         axis=-2)


class MLPDecoder(nn.Module):
    def __init__(self,
                 n_primitives,
                 input_dim,
                 output_dim,
                 factor=1,
                 dim=3,
                 act='leaky',
                 shape_sampler_layer_depth=3,
                 is_feature_radius=True,
                 is_feature_coord=True,
                 is_feature_angles=True):
        super().__init__()
        self.n_primitives = n_primitives
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dim = dim
        hidden_size = int(128 // factor)
        self.is_feature_angles = is_feature_angles
        self.is_feature_coord = is_feature_coord
        self.is_feature_radius = is_feature_radius
        self.shape_sampler_layer_depth = shape_sampler_layer_depth

        self.periodic_input_dim = (self.dim -
                                   1) * 2 if self.is_feature_angles else 0
        self.periodic_input_dim += self.dim if self.is_feature_coord else 0
        self.periodic_input_dim += 1 if self.is_feature_radius else 0
        """
        multires = 10
        embed_kwargs = {
            'include_input': True,
            'input_dims': self.dim,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.cos, torch.sin]
        }
        freq_bands = 2.**torch.linspace(0., embed_kwargs['max_freq_log2'],
                                        embed_kwargs['num_freqs'])
        # B, N, dim * P
        self.freq_bands = freq_bands.view(1, 1, -1, 1).to('cuda')
        print('periodic_input_dim', self.periodic_input_dim)
        self.embedder = Embedder(**embed_kwargs)
        self.periodic_input_dim *= (multires * 2 + 1)
        print('periodic_input_dim', self.periodic_input_dim)
        """

        self.in_conv1d = nn.Conv1d(input_dim, hidden_size, 1)
        self.periodic_in_conv1d = nn.Conv1d(
            self.periodic_input_dim * self.n_primitives, hidden_size, 1)
        self.act = nn.LeakyReLU(0.2, True)
        self.out_conv1d = nn.Conv1d(hidden_size, n_primitives * output_dim, 1)
        #self.bn = nn.BatchNorm1d(hidden_size)

        self.blocks = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, 1)
            for _ in range(shape_sampler_layer_depth)
        ])
        self.r = torch.rand([3, 3, 100, 1]).float() * 2

    def forward(self, feature, thetas, radius, coord, *args, **kwargs):
        # B, 1 or N, P, dim - 1
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert feature.shape[0] == thetas.shape[0]
        B, N, P, Dn1 = thetas.shape
        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        _, feature_dim = feature.shape
        assert feature_dim == self.input_dim
        # B, 1 or N, dim - 1, P
        thetas_transposed = thetas.transpose(2, 3)  #.contiguous()

        radius_transposed = radius.view(B, self.n_primitives, 1,
                                        P)  #.contiguous()

        sin = thetas_transposed.sin()
        cos = thetas_transposed.cos()

        # B, N, dim - 1, P
        sincos = (torch.cat([sin, cos], axis=-2) * 100).expand(
            -1, self.n_primitives, -1, -1)

        # Feature dims has to be (B, P, self.n_primitives, feature dim)
        periodic_feature_list = []
        if self.is_feature_angles:
            periodic_feature_list.append(sincos)

        if self.is_feature_radius:
            periodic_feature_list.append(radius_transposed)

        if self.is_feature_coord:
            transposed_target = coord.transpose(2, 3)  #.contiguous()
            periodic_feature_list.append(transposed_target)
        #for f in feature_list:
        #    print(f.shape)

        # B, N, feature_dim, P
        periodic_feature = torch.cat(periodic_feature_list,
                                     axis=-2).contiguous()
        """
        #periodic_feature = self.embedder.embed(periodic_feature)

        periodic_feature_raw = (
            self.freq_bands *
            periodic_feature.view(B, self.n_primitives, 1, -1) / 3)
        periodic_feature_raw = periodic_feature_raw.view(
            B, self.n_primitives, int(
                (self.periodic_input_dim - self.dim) / 2), P)
        periodic_feature_sin = torch.sin(periodic_feature_raw)
        periodic_feature_cos = torch.cos(periodic_feature_raw)
        periodic_feature = torch.cat(
            [periodic_feature_sin, periodic_feature_cos, periodic_feature],
            axis=-2)
        """

        # B, N, input_dim, P
        feature = self.in_conv1d(feature.view(B, self.input_dim, 1))
        periodic_feature = self.periodic_in_conv1d(
            periodic_feature.view(B,
                                  self.n_primitives * self.periodic_input_dim,
                                  P))

        net = feature + periodic_feature
        net = periodic_feature

        for block in self.blocks:
            net = block(net)

        radius = self.out_conv1d(
            self.act(net)).view(B, self.n_primitives, self.output_dim,
                                P).transpose(2, 3).contiguous()

        # B, n_primitives, P, self.label_num = 1
        print(coord.mean(), thetas.mean())
        return radius

    def to(self, device):
        super().to(device)
        self.freq_bands = self.freq_bands.to(device)
        return self


class PrimitiveWiseGroupConvDecoderLegacy(nn.Module):
    def __init__(self,
                 n_primitives,
                 input_dim,
                 output_dim,
                 factor=1,
                 dim=3,
                 act='leaky',
                 shape_sampler_layer_depth=3,
                 is_feature_radius=True,
                 is_feature_coord=True,
                 is_feature_angles=True):
        super().__init__()
        self.n_primitives = n_primitives
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dim = dim
        hidden_size = int(128 // factor)
        self.is_feature_angles = is_feature_angles
        self.is_feature_coord = is_feature_coord
        self.is_feature_radius = is_feature_radius
        self.shape_sampler_layer_depth = shape_sampler_layer_depth

        self.periodic_input_dim = (self.dim -
                                   1) * 2 if self.is_feature_angles else 0
        self.periodic_input_dim += self.dim if self.is_feature_coord else 0
        self.periodic_input_dim += 1 if self.is_feature_radius else 0

        self.in_conv1d = nn.Conv1d(
            (input_dim + self.periodic_input_dim) * self.n_primitives,
            hidden_size * self.n_primitives,
            1,
            groups=self.n_primitives)
        self.act = nn.LeakyReLU(0.2, True)
        self.out_conv1d = nn.Conv1d(hidden_size * self.n_primitives,
                                    n_primitives * output_dim,
                                    1,
                                    groups=self.n_primitives)
        #self.bn = nn.BatchNorm1d(hidden_size)

        self.blocks = [
            nn.Conv1d(hidden_size * self.n_primitives,
                      hidden_size * self.n_primitives,
                      1,
                      groups=self.n_primitives)
            for _ in range(shape_sampler_layer_depth)
        ]
        self.r = torch.rand([3, 3, 100, 1]).float() * 2

    def forward(self, feature, thetas, radius, coord, *args, **kwargs):
        # B, 1 or N, P, dim - 1
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert feature.shape[0] == thetas.shape[0]
        B, N, P, Dn1 = thetas.shape
        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        _, feature_dim = feature.shape
        assert feature_dim == self.input_dim
        # B, P, 1 or N, dim - 1
        thetas_transposed = thetas.transpose(1, 2)  #.contiguous()

        radius_transposed = radius.view(B, self.n_primitives, P,
                                        1).transpose(1, 2)  #.contiguous()

        sin = thetas_transposed.sin()
        cos = thetas_transposed.cos()

        # B, P, N, dim - 1
        sincos = (torch.cat([sin, cos], axis=-1) * 100).expand(
            -1, -1, self.n_primitives, -1)

        # Feature dims has to be (B, P, self.n_primitives, feature dim)
        periodic_feature_list = []
        if self.is_feature_angles:
            periodic_feature_list.append(sincos)

        if self.is_feature_radius:
            periodic_feature_list.append(radius_transposed)

        if self.is_feature_coord:
            transposed_target = coord.transpose(1, 2)  #.contiguous()
            periodic_feature_list.append(transposed_target)

        # B, P, N, input_dim
        feature = feature.view(B, 1, 1,
                               self.input_dim).expand(-1, P, self.n_primitives,
                                                      -1)
        periodic_feature_list.append(feature)
        # B, P, N, feature_dim
        feature = torch.cat(periodic_feature_list, axis=-1).contiguous()

        _, _, _n, _f = feature.shape
        net = self.in_conv1d(feature.view(B * P, _n * _f, 1))

        #for block in self.blocks:
        #    net = block(net)
        net = self.blocks[0](net)

        radius = self.out_conv1d(
            self.act(net)).view(B, P, self.n_primitives,
                                self.output_dim).transpose(1, 2).contiguous()

        # B, n_primitives, P, self.label_num = 1
        print(coord.mean(), thetas.mean())
        return radius


class BatchNormDecoderSharedWeight(nn.Module):
    def __init__(self,
                 n_primitives,
                 input_dim,
                 output_dim,
                 factor=1,
                 dim=3,
                 act='leaky',
                 shape_sampler_layer_depth=3,
                 is_feature_radius=True,
                 is_feature_coord=True,
                 is_feature_angles=True):
        super().__init__()
        self.n_primitives = n_primitives
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dim = dim
        hidden_size = int(128 // factor)
        self.is_feature_angles = is_feature_angles
        self.is_feature_coord = is_feature_coord
        self.is_feature_radius = is_feature_radius
        self.shape_sampler_layer_depth = shape_sampler_layer_depth

        self.periodic_input_dim = (self.dim -
                                   1) * 2 if self.is_feature_angles else 0
        self.periodic_input_dim += self.dim if self.is_feature_coord else 0
        self.periodic_input_dim += 1 if self.is_feature_radius else 0

        self.in_conv1d = nn.Conv2d(input_dim, hidden_size, 1)
        self.periodic_in_conv1d = nn.Conv2d(self.periodic_input_dim, input_dim,
                                            1)
        self.act = nn.LeakyReLU(0.2, True)
        self.out_conv1d = nn.Conv2d(hidden_size, output_dim, 1)
        self.bn = nn.BatchNorm2d(hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockConv2d(hidden_size)
            for _ in range(self.shape_sampler_layer_depth)
        ])

    def forward(self, feature, thetas, radius, coord, *args, **kwargs):
        # B, 1 or N, P, dim - 1
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert feature.shape[0] == thetas.shape[0]
        B, N, P, Dn1 = thetas.shape
        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        _, feature_dim = feature.shape
        assert feature_dim == self.input_dim

        if self.is_feature_angles:
            raise ValueError('angle feature not supported')

        if self.is_feature_radius:
            raise ValueError('radius feature not supported')

        if not self.is_feature_coord:
            raise ValueError('coord feature must be specified')

        # B, dim, N, P
        transposed_coord = coord.permute(0, 3, 1, 2).contiguous()
        periodic_feature = self.periodic_in_conv1d(transposed_coord)
        net = periodic_feature + feature.view(B, -1, 1, 1)

        #for block in self.blocks:
        #    net = block(net)
        net = self.in_conv1d(net)
        net1 = self.blocks[0](net)
        net2 = self.blocks[1](net1)
        net3 = self.blocks[2](net2)

        radius = self.out_conv1d(self.act(self.bn(net3))).view(
            B, self.output_dim, self.n_primitives, P).permute(0, 2, 3,
                                                              1).contiguous()

        # B, n_primitives, P, self.label_num = 1
        return radius

    def to(self, device):
        super().to(device)
        #self.freq_bands = self.freq_bands.to(device)
        return self


decoder_dict = {
    'PrimitiveWiseGroupConvDecoder': PrimitiveWiseGroupConvDecoder,
    'PrimitiveWiseGroupConvDecoderLegacy': PrimitiveWiseGroupConvDecoderLegacy,
    'PrimitiveWiseGroupConvDecoderWithNoEncoder':
    PrimitiveWiseGroupConvDecoderWithNoEncoder,
    'PrimitiveWiseGroupConvDecoderUseFCToExpandFeature':
    PrimitiveWiseGroupConvDecoderUseFCToExpandFeature,
    'ShapeDecoderCBatchNorm': ShapeDecoderCBatchNorm,
    'BatchNormDecoder': BatchNormDecoder,
    'BatchNormDecoderSharedWeight': BatchNormDecoderSharedWeight,
    'MLPDecoder': MLPDecoder
}
