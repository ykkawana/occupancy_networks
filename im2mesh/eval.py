# from im2mesh import icp
import logging
import random
import numpy as np
import trimesh
# from scipy.spatial import cKDTree
from im2mesh.utils.libkdtree import KDTree
from im2mesh.utils.libmesh import check_mesh_contains
from im2mesh.common import compute_iou
from pykeops.torch import LazyTensor
import kaolin as kal
import torch
import warnings

# Maximum values for bounding box [-0.5, 0.5]^3
EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer': 6,
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': -1.,
    'normals accuracy': -1.,
    'normals': -1.,
}

logger = logging.getLogger(__name__)


class MeshEvaluator(object):
    ''' Mesh evaluation class.

    It handles the mesh evaluation process.

    Args:
        n_points (int): number of points to be used for evaluation
    '''
    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self,
                  mesh,
                  pointcloud_tgt,
                  normals_tgt,
                  points_iou,
                  occ_tgt,
                  is_eval_explicit_mesh=False,
                  vertex_visibility=None):
        ''' Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        '''
        if is_eval_explicit_mesh:
            assert vertex_visibility is not None
            sampled_vertex_idx = np.zeros_like(vertex_visibility).astype(
                np.bool)
            pointcloud = mesh.vertices[vertex_visibility, :]
            normals = mesh.vertex_normals[vertex_visibility, :]
            pointcloud = pointcloud.astype(np.float32)
            normals = normals.astype(np.float32)

            if pointcloud.shape[0] > self.n_points:
                select_idx = random.sample(range(pointcloud.shape[0]),
                                           self.n_points)
                pointcloud = pointcloud[select_idx:]
            if normals.shape[0] > self.n_points:
                select_idx = random.sample(range(normals.shape[0]),
                                           self.n_points)
                normals = normals[select_idx:]
            if pointcloud_tgt.shape[0] > pointcloud.shape[0]:
                select_idx = random.sample(range(pointcloud.shape[0]),
                                           pointcloud.shape[0])
                pointcloud_tgt = pointcloud_tgt[select_idx, :]
            if normals_tgt.shape[0] > normals.shape[0]:
                select_idx = random.sample(range(normals.shape[0]),
                                           pointcloud.shape[0])
                normals_tgt = normals_tgt[select_idx, :]
        else:
            if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
                pointcloud, idx = mesh.sample(self.n_points, return_index=True)
                pointcloud = pointcloud.astype(np.float32)
                normals = mesh.face_normals[idx]
            else:
                pointcloud = np.empty((0, 3))
                normals = np.empty((0, 3))

        out_dict = self.eval_pointcloud(pointcloud, pointcloud_tgt, normals,
                                        normals_tgt)

        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            occ = check_mesh_contains(mesh, points_iou)
            out_dict['iou'] = compute_iou(occ, occ_tgt)
        else:
            out_dict['iou'] = 0.

        return out_dict

    def eval_pointcloud(self,
                        pointcloud,
                        pointcloud_tgt,
                        normals=None,
                        normals_tgt=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            logger.warn('Empty pointcloud / mesh detected!')
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals)
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(pointcloud, normals,
                                                  pointcloud_tgt, normals_tgt)
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (0.5 * completeness_normals +
                               0.5 * accuracy_normals)
        chamferL1 = 0.5 * (completeness + accuracy)

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
        }

        return out_dict

    def eval_fscore_from_mesh(self,
                              mesh,
                              pointcloud_tgt,
                              thresholds,
                              is_eval_explicit_mesh=False,
                              vertex_visibility=None):
        ''' Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        '''

        if is_eval_explicit_mesh:
            assert vertex_visibility is not None
            pointcloud = mesh.vertices[vertex_visibility, :]
            pointcloud = pointcloud.astype(np.float32)

            if pointcloud.shape[0] > self.n_points:
                select_idx = random.sample(range(pointcloud.shape[0]),
                                           self.n_points)
                pointcloud = pointcloud[select_idx:]

            if pointcloud_tgt.shape[0] > pointcloud.shape[0]:
                select_idx = random.sample(range(pointcloud.shape[0]),
                                           pointcloud.shape[0])
                pointcloud_tgt = pointcloud_tgt[select_idx, :]

        else:
            if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
                pointcloud, idx = mesh.sample(self.n_points, return_index=True)
                pointcloud = pointcloud.astype(np.float32)
            else:
                pointcloud = np.empty((0, 3))
        """
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            pointcloud = mesh.sample(self.n_points, return_index=False)
            pointcloud = pointcloud.astype(np.float32)
        else:
            pointcloud = np.empty((0, 3))
        """

        out_dict = fscore(pointcloud[np.newaxis, ...],
                          pointcloud_tgt[np.newaxis, ...],
                          thresholds=thresholds,
                          mode='pykeops')
        if out_dict is None:
            return out_dict
        else:
            out_dict = {
                k: v[0].item()
                for k, v in out_dict.items() if v is not None
            }

        return out_dict

    def eval_fscore_from_mesh_batch(self, pointcloud_pred, pointcloud_tgt,
                                    thresholds):
        ''' Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        '''
        out_dict = fscore(pointcloud_pred,
                          pointcloud_tgt,
                          thresholds=thresholds,
                          mode='pykeops')

        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0],
                                       dtype=np.float32)
    return dist, normals_dot_product


def distance_p2m(points, mesh):
    ''' Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    '''
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def chamfer_distance(pred, target, pykeops=True):
    assert pykeops
    # B, P, 1, dim
    pred_lazy = LazyTensor(pred.unsqueeze(2))
    # B, 1, P2, dim
    target_lazy = LazyTensor(target.unsqueeze(1))

    # B, P, P2, dim
    dist = (pred_lazy - target_lazy).norm2()

    # B, P, dim
    pred2target = dist.min(2).squeeze(-1)

    # B, P2, dim
    target2pred = dist.min(1).squeeze(-1)

    return pred2target, target2pred


def fscore(pred_points, target_points, thresholds=[0.01], mode='pykeops'):
    assert mode in ['kaolin', 'pykeops']
    assert isinstance(thresholds, list)

    if isinstance(pred_points, np.ndarray):
        pred_points = torch.from_numpy(pred_points).to('cuda')
    if isinstance(target_points, np.ndarray):
        target_points = torch.from_numpy(target_points).to('cuda')
    assert pred_points.ndim == 3 and target_points.ndim == 3
    assert len(pred_points.shape) == 3 and len(
        target_points.shape) == 3, (pred_points.shape, target_points.shape,
                                    len(pred_points.shape),
                                    len(target_points.shape))

    try:
        assert pred_points.shape[1] == target_points.shape[
            1], pred_points.shape[1] == target_points.shape[1]
    except:
        warnings.warn('point shapes are not same!')
        return {}

    f_scores = {}
    if mode == 'kaolin':
        assert False
        for threshold in thresholds:
            b_f = []
            for idx in range(target_points.shape[0]):
                t = target_points[idx]
                p = pred_points[idx]
                f_score = kal.metrics.f_score(t, p, radius=threshold)
                b_f.append(f_score)
            f_scores['fscore_th={}'.format(threshold)] = torch.stack(
                b_f, 0).detach().to('cpu')

    elif mode == 'pykeops':

        gt_distances, pred_distances = chamfer_distance(
            pred_points, target_points)

        for threshold in thresholds:
            fn = (pred_distances > threshold).sum(-1).float()
            fp = (gt_distances > threshold).sum(-1).float()
            tp = (gt_distances <= threshold).sum(-1).float()

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            f_scores['fscore_th={}'.format(threshold)] = f_score.detach().to(
                'cpu')
    else:
        raise NotImplementedError
    return f_scores
