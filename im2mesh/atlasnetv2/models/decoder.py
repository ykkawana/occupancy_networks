import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (ResnetBlockFC, CResnetBlockConv1d, CBatchNorm1d,
                            CBatchNorm1d_legacy, ResnetBlockConv1d)

from atlasnetv2.auxiliary.model import mlpAdj, patchDeformationMLP

from periodic_shapes.layers import primitive_wise_layers


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
        #self.decoder = PatchDeformGroupWiseMLPAdjInOcc(self.options)
        self.decoder = PatchDeformMLPAdjInOcc(self.options)

    def forward(self, p, z, color_feature, grid=None, **kwargs):
        # grid (B, P, dim) -> (B, dim, P)
        transposed_grid = grid[:, 0, :, :].transpose(-1, -2)
        out, _ = self.decoder(color_feature, transposed_grid)
        #
        B = transposed_grid.shape[0]
        out = out.view(B, self.options.npatch, -1, 3)
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


class PatchDeformGroupWiseMLPAdjInOcc(nn.Module):
    """Atlas net auto encoder"""
    def __init__(self, options):

        super(PatchDeformGroupWiseMLPAdjInOcc, self).__init__()

        self.npatch = options.npatch
        self.nlatent = options.nlatent
        self.patchDim = options.patchDim
        assert self.patchDim == 2
        self.patchDeformDim = options.patchDeformDim

        #encoder decoder and patch deformation module
        #==============================================================================
        self.decoder = GroupWisemlpAdj(nlatent=self.patchDeformDim +
                                       self.nlatent,
                                       npatch=self.npatch)
        self.patchDeformation = patchDeformationGroupWiseMLP(
            patchDim=self.patchDim,
            patchDeformDim=self.patchDeformDim,
            npatch=self.npatch)
        #==============================================================================

    def forward(self, x, grid):

        #encoder
        #==============================================================================
        #x = self.encoder(x.transpose(2, 1).contiguous())
        #==============================================================================

        # B, P, dims
        grid[:, 2:, :] = 0
        # B, N, dims, P
        rand_grid1 = grid.unsqueeze(1).expand(-1, self.npatch, -1,
                                              -1).contiguous()

        #random planar patch
        #==========================================================================
        rand_grid2 = self.patchDeformation(rand_grid1)
        #==========================================================================

        #cat with latent vector and decode
        #==========================================================================
        # B, nlatent -> B, N, nlatent, P
        y1 = x.view(x.size(0), 1, x.size(1), 1).expand(-1, self.npatch, -1,
                                                       rand_grid2.size(3))
        y2 = torch.cat([y1, rand_grid2], axis=2).contiguous()

        # B, N, defromdim, P
        out = self.decoder(y2)
        #==========================================================================

        #B, N, P, deformdim
        return out.transpose(3, 2).contiguous(), None


class patchDeformationGroupWiseMLP(nn.Module):
    """deformation of a 2D patch into a 3D surface"""
    def __init__(self, patchDim=2, patchDeformDim=3, npatch=16, tanh=True):

        super(patchDeformationGroupWiseMLP, self).__init__()
        layer_size = 128
        self.tanh = tanh
        #self.conv1 = torch.nn.Conv1d(patchDim, layer_size, 1)
        #self.conv2 = torch.nn.Conv1d(layer_size, layer_size, 1)
        #self.conv3 = torch.nn.Conv1d(layer_size, patchDeformDim, 1)
        self.bn1 = torch.nn.BatchNorm1d(layer_size * npatch)
        self.bn2 = torch.nn.BatchNorm1d(layer_size * npatch)

        self.conv1 = primitive_wise_layers.PrimitiveWiseLinear(npatch,
                                                               patchDim,
                                                               layer_size,
                                                               act='none')
        self.conv2 = primitive_wise_layers.PrimitiveWiseLinear(npatch,
                                                               layer_size,
                                                               layer_size,
                                                               act='none')
        self.conv3 = primitive_wise_layers.PrimitiveWiseLinear(npatch,
                                                               layer_size,
                                                               patchDeformDim,
                                                               act='none')
        self.th = nn.Tanh()

    def forward(self, x):
        B, N, _, P = x.shape
        x = F.relu(self.bn1(self.conv1(x).view(B, -1, P))).view(B, N, -1, P)
        x = F.relu(self.bn2(self.conv2(x).view(B, -1, P))).view(B, N, -1, P)
        if self.tanh:
            x = self.th(self.conv3(x))
        else:
            x = self.conv3(x)
        return x


class GroupWisemlpAdj(nn.Module):
    def __init__(self, nlatent=1024, npatch=16):
        """Atlas decoder"""

        super(GroupWisemlpAdj, self).__init__()
        self.nlatent = nlatent
        self.conv1 = primitive_wise_layers.PrimitiveWiseLinear(npatch,
                                                               self.nlatent,
                                                               self.nlatent,
                                                               act='none')
        self.conv2 = primitive_wise_layers.PrimitiveWiseLinear(npatch,
                                                               self.nlatent,
                                                               self.nlatent //
                                                               2,
                                                               act='none')
        self.conv3 = primitive_wise_layers.PrimitiveWiseLinear(
            npatch, self.nlatent // 2, self.nlatent // 4, act='none')
        self.conv4 = primitive_wise_layers.PrimitiveWiseLinear(npatch,
                                                               self.nlatent //
                                                               4,
                                                               3,
                                                               act='none')

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.nlatent * npatch)
        self.bn2 = torch.nn.BatchNorm1d(self.nlatent // 2 * npatch)
        self.bn3 = torch.nn.BatchNorm1d(self.nlatent // 4 * npatch)

    def forward(self, x):
        B, N, _, P = x.shape
        x = F.relu(self.bn1(self.conv1(x).view(B, -1, P))).view(B, N, -1, P)
        x = F.relu(self.bn2(self.conv2(x).view(B, -1, P))).view(B, N, -1, P)
        x = F.relu(self.bn3(self.conv3(x).view(B, -1, P))).view(B, N, -1, P)
        x = self.th(self.conv4(x).view(B, -1, P)).view(B, N, -1, P)
        return x
