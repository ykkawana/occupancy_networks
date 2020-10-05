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
                 is_gen_primitive_wise_watertight_mesh=False,
                 is_gen_surface_points=False,
                 is_gen_primitive_wise_watertight_mesh_debugged=False,
                 is_gen_integrated_mesh=False,
                 is_gen_implicit_mesh=False,
                 is_gen_skip_vertex_attributes=False,
                 is_just_measuring_time=False,
                 is_skip_realign=False,
                 is_fit_to_gt_loc_scale=False,
                 is_gen_whole_mesh=True,
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
        self.is_gen_primitive_wise_watertight_mesh = is_gen_primitive_wise_watertight_mesh
        self.is_gen_surface_points = is_gen_surface_points
        self.is_gen_primitive_wise_watertight_mesh_debugged = is_gen_primitive_wise_watertight_mesh_debugged
        self.is_gen_skip_vertex_attributes = is_gen_skip_vertex_attributes
        self.is_gen_integrated_mesh = is_gen_integrated_mesh
        self.is_just_measuring_time = is_just_measuring_time
        self.is_skip_realign = is_skip_realign
        self.is_fit_to_gt_loc_scale = is_fit_to_gt_loc_scale
        self.is_gen_whole_mesh = is_gen_whole_mesh
        self.is_gen_implicit_mesh = is_gen_implicit_mesh

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

        if not self.is_skip_realign:
            imnet_pointcloud = data.get('pointcloud.imnet_points').to(
                device).float()
            imnet_pointcloud = self.gen_helper.roty90(imnet_pointcloud,
                                                      inv=True)

        kwargs = {}

        # Encode inputs

        if self.is_gen_implicit_mesh:

            out_m, t = self.gen_helper.encode(inputs, measure_time=True)
            z = None
            mesh = self.generate_from_latent(z,
                                             out_m,
                                             stats_dict=stats_dict,
                                             **kwargs)
            verts = torch.from_numpy(mesh.vertices).float().to(
                self.device).unsqueeze(0)
            verts = self.gen_helper.roty90(verts, inv=True)
            if not self.is_skip_realign:
                verts = utils.realign(verts, imnet_pointcloud, occ_pointcloud)

            verts = verts[0].cpu().numpy()

            mesh.vertices = verts

            stats_dict['time (encode inputs)'] = 0.
            if return_stats:
                return mesh, stats_dict, None, None, None
            else:
                return mesh, None, None, None

        else:
            with torch.no_grad():
                t0 = time.time()
                out_m, t = self.gen_helper.encode(inputs, measure_time=True)
                stats_dict['time (encode inputs)'] = t

                t0 = time.time()
                model_float, t = self.gen_helper.eval_points(out_m,
                                                             measure_time=True)
                stats_dict['time (eval points)'] = t

                t0 = time.time()
                if self.is_gen_primitive_wise_watertight_mesh:
                    mesh, t = self.gen_helper.gen_primitive_wise_watertight_mesh(
                        model_float, out_m, measure_time=True)

                    if mesh == None:
                        print('invalid convex generation')
                        if return_stats:
                            return None, {}, None, None, None
                        else:
                            return None, None, None, None

                    stats_dict['time (Generate mesh from convex)'] = t
                elif self.is_gen_primitive_wise_watertight_mesh_debugged:
                    mesh, t = self.gen_helper.gen_primitive_wise_watertight_mesh_debugged(
                        model_float, out_m, measure_time=True)

                    breakpoint
                    if mesh == None:
                        print('invalid convex generation')
                        if return_stats:
                            return None, {}, None, None, None
                        else:
                            return None, None, None, None

                    stats_dict['time (Generate mesh from convex)'] = t
                elif self.is_gen_whole_mesh:
                    mesh, t = self.gen_helper.gen_whole_mesh(model_float,
                                                             out_m,
                                                             measure_time=True)

                    if mesh == None:
                        print('invalid convex generation')
                        if return_stats:
                            return None, {}, None, None, None
                        else:
                            return None, None, None, None

                    stats_dict['time (Generate mesh from convex)'] = t
                elif self.is_gen_integrated_mesh:
                    mesh, t = self.gen_helper.gen_integrated_mesh(
                        model_float, out_m, measure_time=True)
                    stats_dict['time (Generate mesh from convex)'] = t

                elif self.is_gen_surface_points:
                    verts, t = self.gen_helper.gen_surface_points(
                        model_float, out_m, measure_time=True)
                    stats_dict['time (Generate mesh from convex)'] = t
                    mesh = trimesh.Trimesh(verts)

                else:
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

            if self.is_just_measuring_time:
                if return_stats:
                    return mesh, stats_dict, None, None, None
                else:
                    return mesh, None, None, None

            verts_pos = []
            if isinstance(mesh, list):
                meshes = mesh
                vertss = []
                for m in meshes:
                    verts_pos.append(len(m.vertices))
                    vertss.append(m.vertices)
                mesh = trimesh.Trimesh(np.concatenate(vertss))
            verts = torch.from_numpy(mesh.vertices).float().to(
                self.device).unsqueeze(0)
            verts = self.gen_helper.roty90(verts, inv=True)
            if not self.is_skip_realign:
                verts = utils.realign(verts, imnet_pointcloud, occ_pointcloud)

            if self.is_fit_to_gt_loc_scale:
                verts = utils.realign(verts,
                                      verts,
                                      occ_pointcloud,
                                      adjust_bbox=True)

            verts = verts[0].cpu().numpy()
            if len(verts_pos) > 0:
                cpos = 0
                ms = []
                for m, vpos in zip(meshes, verts_pos):
                    m.vertices = verts[cpos:cpos + vpos, :]
                    cpos += vpos
                    ms.append(m)
                mesh = ms

            else:
                mesh.vertices = verts

            if self.is_gen_skip_vertex_attributes:
                if return_stats:
                    return mesh, stats_dict, None, None, None
                else:
                    return mesh, None, None, None
            elif self.is_gen_surface_points:
                if return_stats:
                    return mesh, stats_dict, mesh.vertices, None, None
                else:
                    return mesh, None, mesh.vertices, None
            # (P, dim), (P, dim), (P,)
            with torch.no_grad():
                points, normal, visibility = self.gen_helper.sample_primitive_agnostic_surface_points(
                    out_m)

            perm_idx = torch.randperm(points.shape[0])
            points = points[perm_idx, :]
            normal = normal[perm_idx, :]
            visibility = visibility[perm_idx]

            points = self.gen_helper.roty90(points.unsqueeze(0), inv=True)
            if not self.is_skip_realign:
                points = utils.realign(points, imnet_pointcloud,
                                       occ_pointcloud)

            if self.is_fit_to_gt_loc_scale:
                points = utils.realign(points,
                                       points,
                                       occ_pointcloud,
                                       adjust_bbox=True)

            points = points[0].cpu().numpy()

            normal = self.gen_helper.roty90(normal.unsqueeze(0),
                                            inv=True)[0].cpu().numpy()
            visibility = visibility.cpu().numpy()
            if return_stats:
                return mesh, stats_dict, points, normal, visibility
            else:
                return mesh, points, normal, visibility

    def generate_from_latent(self, z, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = 0.99
        #threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid((-0.5, ) * 3, (0.5, ) * 3,
                                              (nx, ) * 3)
            values = self.eval_points(pointsf, z, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(self.resolution0, self.upsampling_steps,
                                  threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(pointsf, z, c,
                                          **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, z, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat, t = self.gen_helper.eval_input_points(
                    pi, c, measure_time=True)

            occ_hats.append(occ_hat[0, :, 0].detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, z, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = 0.99
        #threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(occ_hat_padded,
                                                       threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles, process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, z, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

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
        threshold = 0.99
        #threshold = np.log(self.threshold) - np.log(1. - self.threshold)

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
