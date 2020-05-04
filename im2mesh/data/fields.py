import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from im2mesh.data.core import Field
from im2mesh.utils import binvox_rw
from periodic_shapes import utils
from im2mesh.atlasnetv2 import utils as atv2_utils
import torch


class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


class CategoryField(Field):
    ''' Basic category field.'''
    def load(self, model_path, idx, category):
        ''' Loads the category field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return category

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


class ImagesField(Field):
    ''' Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''
    def __init__(self,
                 folder_name,
                 transform=None,
                 cfg=None,
                 extension='jpg',
                 random_view=True,
                 with_camera=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera
        self.cfg = cfg

        self.is_bspnet = False
        if cfg is not None and self.cfg['method'] == 'bspnet':
            self.is_bspnet = True
            assert 'bspnet' in self.cfg['data']
            bspnet_config = self.cfg['data']['bspnet']
            assert bspnet_config['path'].startswith('data')

            assert not self.with_camera

            self.extension = bspnet_config['extension']
            self.folder_name = bspnet_config['img_folder']

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.is_bspnet:
            model_path = model_path.replace(self.cfg['data']['path'],
                                            self.cfg['data']['bspnet']['path'])

        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        if self.random_view:
            idx_img = random.randint(0, len(files) - 1)
        else:
            idx_img = 0
        filename = files[idx_img]

        if self.is_bspnet:

            image = np.load(filename)['image']
            # 1 x 128 x 128
            image = torch.from_numpy(image)
        else:
            image = Image.open(filename).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

        data = {None: image}

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete


# 3D Fields
class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self,
                 file_name,
                 transform=None,
                 with_transforms=False,
                 unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.with_transforms and 'loc' in data and 'scale' in data:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data


class VoxelsField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''
    def __init__(self,
                 file_name,
                 transform=None,
                 cfg=None,
                 with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.cfg = cfg

        self.is_bspnet = False
        if cfg is not None and self.cfg['method'] == 'bspnet':
            self.is_bspnet = True
            assert 'bspnet' in self.cfg['data']
            bspnet_config = self.cfg['data']['bspnet']
            assert bspnet_config['path'].startswith('data')
            assert 'pointcloud_file' in bspnet_config

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            None: points,
            'normals': normals,
        }

        if self.is_bspnet:
            bsp_model_path = model_path.replace(
                self.cfg['data']['path'], self.cfg['data']['bspnet']['path'])

            data['imnet_points'] = torch.from_numpy(
                trimesh.load(
                    os.path.join(bsp_model_path, self.cfg['data']['bspnet']
                                 ['pointcloud_file'])).vertices)

        if self.with_transforms and 'loc' in data and 'scale' in data:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


# NOTE: this will produce variable length output.
# You need to specify collate_fn to make it work with a data laoder
class MeshField(Field):
    ''' Mesh field.

    It provides the field used for mesh data. Note that, depending on the
    dataset, it produces variable length output, so that you need to specify
    collate_fn to make it work with a data loader.

    Args:
        file_name (str): file name
        transform (list): list of transforms applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        mesh = trimesh.load(file_path, process=False)
        if self.transform is not None:
            mesh = self.transform(mesh)

        data = {
            'verts': mesh.vertices,
            'faces': mesh.faces,
        }

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class SphericalCoordinateField(Field):
    ''' Angle field class.

    It provides the class used for spherical coordinate data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self,
                 primitive_points_sample_n,
                 mode,
                 is_normal_icosahedron=False,
                 is_normal_uv_sphere=False,
                 icosahedron_subdiv=2,
                 icosahedron_uv_margin=1e-5,
                 icosahedron_uv_margin_phi=1e-5,
                 uv_sphere_length=20,
                 normal_mesh_no_invert=False,
                 *args,
                 **kwargs):
        self.primitive_points_sample_n = primitive_points_sample_n
        self.mode = mode
        self.is_normal_icosahedron = is_normal_icosahedron
        self.is_normal_uv_sphere = is_normal_uv_sphere
        self.icosahedron_uv_margin = icosahedron_uv_margin
        self.icosahedron_uv_margin_phi = icosahedron_uv_margin_phi

        if self.is_normal_icosahedron:
            icosamesh = trimesh.creation.icosphere(
                subdivisions=icosahedron_subdiv)

            if not normal_mesh_no_invert:
                icosamesh.invert()
            uv = trimesh.util.vector_to_spherical(icosamesh.vertices)

            uv[:, 1] = uv[:, 1] - np.pi / 2

            uv_thetas = torch.tensor(uv).float()
            uv_th = uv_thetas[:, 0].clamp(min=(-np.pi + icosahedron_uv_margin),
                                          max=(np.pi - icosahedron_uv_margin))
            uv_ph = uv_thetas[:, 1].clamp(
                min=(-np.pi / 2 + icosahedron_uv_margin),
                max=(np.pi / 2 - icosahedron_uv_margin))
            self.angles_for_nomal = torch.stack([uv_th, uv_ph], axis=-1)
            self.face_for_normal = torch.from_numpy(icosamesh.faces)

        elif self.is_normal_uv_sphere:
            thetas = utils.sample_spherical_angles(
                batch=1,
                sample_num=uv_sphere_length,
                sampling='grid',
                device='cpu',
                dim=3,
                sgn_convertible=True,
                phi_margin=icosahedron_uv_margin_phi,
                theta_margin=icosahedron_uv_margin)
            mesh = trimesh.creation.uv_sphere(
                theta=np.linspace(0, np.pi, uv_sphere_length),
                phi=np.linspace(-np.pi, np.pi, uv_sphere_length))
            #thetas = torch.where(thetas.abs() < icosahedron_uv_margin_phi,
            #                     torch.tensor([icosahedron_uv_margin_phi]),
            #                     thetas)
            if not normal_mesh_no_invert:
                mesh.invert()
            self.angles_for_nomal = thetas[0]
            self.face_for_normal = torch.from_numpy(mesh.faces)

    def load(self, model_path, idx, category):
        ''' Sample spherical coordinate.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        angles = utils.sample_spherical_angles(
            batch=1,
            sample_num=self.primitive_points_sample_n,
            sampling='grid' if self.mode in ['test', 'val'] else 'uniform',
            device='cpu',
            dim=3,  #sgn_convertible=True, phi_margin=1e-5, theta_margin=1e-5)
            sgn_convertible=True,
            phi_margin=self.icosahedron_uv_margin_phi,
            theta_margin=self.icosahedron_uv_margin).squeeze(0)

        data = {None: angles}
        if self.is_normal_icosahedron or self.is_normal_uv_sphere:
            data.update({
                'normal_angles': self.angles_for_nomal.clone(),
                'normal_face': self.face_for_normal.clone()
            })
        return data

    def check_complete(self, _):
        ''' Check if field is complete.

        Returns: True
        '''
        return True


class RawIDField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        category_id, object_id = model_path.split('/')[-2:]
        data = {'category': category_id, 'object': object_id}
        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


# 3D Fields
class SDFPointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['distances']
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'distances': occupancies,
        }

        if self.with_transforms:
            raise ValueError('data for transform not stored')

        if self.transform is not None:
            data = self.transform(data)

        return data


class PlanarPatchField(Field):
    ''' Angle field class.

    It provides the class used for spherical coordinate data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self,
                 mode,
                 patch_side_length=20,
                 is_generate_mesh=False,
                 *args,
                 **kwargs):
        self.mode = mode
        self.patch_side_length = patch_side_length
        self.is_generate_mesh = is_generate_mesh

        if self.is_generate_mesh:
            vertices, faces = atv2_utils.create_planar_mesh(patch_side_length)

            self.vertices = torch.from_numpy(vertices)
            self.faces = torch.from_numpy(faces)

    def load(self, model_path, idx, category):
        ''' Sample spherical coordinate.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        plane_points = utils.generate_grid_samples(
            [0, 1],
            batch=1,
            sample_num=self.patch_side_length,
            sampling='uniform',
            device='cpu',
            dim=2)

        data = {None: plane_points}
        if self.is_generate_mesh:
            data.update({
                'mesh_vertices': self.vertices.clone(),
                'mesh_faces': self.faces.clone()
            })
        return data


class PartLabeledPointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''
    def __init__(self, file_name, cfg, transform=None):
        self.file_name = file_name
        self.transform = transform
        self.shapenet_path = cfg['data']['path']
        self.semseg_shapenet_path = cfg['data']['semseg_path']

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        model_path = model_path.replace(self.shapenet_path,
                                        self.semseg_shapenet_path)
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        labels = pointcloud_dict['labels'].astype(np.float32)

        data = {
            None: points,
            'labels': labels,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True
