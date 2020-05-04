import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
from bspnet import modelSVR
from bspnet import utils
import time


class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (nn.Module): preprocessor for inputs
    '''
    def __init__(self,
                 model,
                 points_batch_size=100000,
                 threshold=0.5,
                 refinement_step=0,
                 device=None,
                 resolution0=16,
                 upsampling_steps=3,
                 with_normals=False,
                 padding=0.1,
                 sample=False,
                 simplify_nfaces=None,
                 preprocessor=None,
                 **kwargs):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor

        self.gen_helper = modelSVR.BSPNetMeshGenerator(self.model,
                                                       device=self.device)

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device

        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        occ_pointcloud = data.get('pointcloud').to(device)

        imnet_pointcloud = data.get('pointcloud.imnet_points').to(
            device).float()
        imnet_pointcloud = self.gen_helper.roty90(imnet_pointcloud, inv=True)

        kwargs = {}

        # Encode inputs
        with torch.no_grad():
            t0 = time.time()
            out_m, t = self.gen_helper.encode(inputs, measure_time=True)
            stats_dict['time (encode inputs)'] = t

            t0 = time.time()
            model_float, t = self.gen_helper.eval_points(out_m,
                                                         measure_time=True)
            stats_dict['time (eval points)'] = t

            t0 = time.time()
            fout2, t = self.gen_helper.gen_mesh(model_float,
                                                out_m,
                                                measure_time=True)

            if fout2 == None:
                print('invalid convex generation')
                if return_stats:
                    return None, {}, None, None, None
                else:
                    return None, None, None, None

            stats_dict['time (Generate mesh from convex)'] = t

            mesh = self.gen_helper.gen_trimesh(fout2)

        verts = torch.from_numpy(mesh.vertices).float().to(
            self.device).unsqueeze(0)
        verts = self.gen_helper.roty90(verts, inv=True)
        verts = utils.realign(verts, imnet_pointcloud, occ_pointcloud)
        verts = verts[0].cpu().numpy()
        mesh.vertices = verts

        # (P, dim), (P, dim), (P,)
        with torch.no_grad():
            points, normal, visibility = self.gen_helper.sample_primitive_agnostic_surface_points(
                out_m)

        perm_idx = torch.randperm(points.shape[0])
        points = points[perm_idx, :]
        normal = normal[perm_idx, :]
        visibility = visibility[perm_idx]

        points = self.gen_helper.roty90(points.unsqueeze(0), inv=True)
        points = utils.realign(points, imnet_pointcloud, occ_pointcloud)
        points = points[0].cpu().numpy()

        normal = self.gen_helper.roty90(normal.unsqueeze(0),
                                        inv=True)[0].cpu().numpy()
        visibility = visibility.cpu().numpy()
        if return_stats:
            return mesh, stats_dict, points, normal, visibility
        else:
            return mesh, points, normal, visibility

    def refine_mesh(self, mesh, occ_hat, z, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert (n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), z, c).logits)
            normal_target = -autograd.grad([face_value.sum()], [face_point],
                                           create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
