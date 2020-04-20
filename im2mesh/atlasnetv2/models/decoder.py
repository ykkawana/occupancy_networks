import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (ResnetBlockFC, CResnetBlockConv1d, CBatchNorm1d,
                            CBatchNorm1d_legacy, ResnetBlockConv1d)

from atlasnetv2.auxiliary.model import mlpAdj, patchDeformationMLP
from atlasnetv2.auxiliary.utils import weights_init


class AtlasNetV2Decoder(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self,
                 dim=3,
                 z_dim=128,
                 c_dim=128,
                 hidden_size=128,
                 leaky=False,
                 **decoder_kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        decoder_kwargs['nlatent'] = hidden_size
        self.options = type('', (), decoder_kwargs)
        self.decoder = PatchDeformMLPAdjInOcc(self.options)
        self.decoder.apply(weights_init)

    def forward(self, p, z, color_feature, grid=None, **kwargs):
        # grid (B, P, dim) -> (B, dim, P)
        transposed_grid = grid[:, 0, :, :].transpose(-1, -2)
        out, _ = self.decoder(color_feature, transposed_grid)

        #
        B = transposed_grid.shape[0]
        out = out.transpose(-1, -2).contiguous().view(B, self.options.npatch,
                                                      -1, 3)
        return out


class PatchDeformMLPAdjInOcc(nn.Module):
    """Atlas net auto encoder"""
    def __init__(self, options):

        super(PatchDeformMLPAdjInOcc, self).__init__()

        self.npatch = options.npatch
        self.nlatent = options.nlatent
        self.patchDim = options.patchDim
        assert self.patchDim == 2
        self.patchDeformDim = options.patchDeformDim

        #encoder decoder and patch deformation module
        #==============================================================================
        self.decoder = nn.ModuleList([
            mlpAdj(nlatent=self.patchDeformDim + self.nlatent)
            for i in range(0, self.npatch)
        ])
        self.patchDeformation = nn.ModuleList(
            patchDeformationMLP(patchDim=self.patchDim,
                                patchDeformDim=self.patchDeformDim)
            for i in range(0, self.npatch))
        #==============================================================================

    def forward(self, x, grid):

        #encoder
        #==============================================================================
        #x = self.encoder(x.transpose(2, 1).contiguous())
        #==============================================================================

        outs = []
        patches = []
        for i in range(0, self.npatch):
            rand_grid = grid

            #random planar patch
            #==========================================================================
            """
            rand_grid = torch.FloatTensor(x.size(0), self.patchDim,
                                          self.npoint // self.npatch).cuda()
            rand_grid.data.uniform_(0, 1)
            """
            rand_grid[:, 2:, :] = 0
            rand_grid = self.patchDeformation[i](rand_grid.contiguous())
            patches.append(rand_grid[0].transpose(1, 0))
            #==========================================================================

            #cat with latent vector and decode
            #==========================================================================
            y = x.unsqueeze(2).expand(x.size(0), x.size(1),
                                      rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
            #==========================================================================

        return torch.cat(outs, 2).transpose(2, 1).contiguous(), patches
