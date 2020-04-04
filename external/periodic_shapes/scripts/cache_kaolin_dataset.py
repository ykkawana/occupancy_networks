import os
import sys
import dotenv
import kaolin as kal
from kaolin.datasets import shapenet 

dotenv.load_dotenv(verbose=True)

category = sys.argv[1]
cache_root = os.getenv('SHAPENET_KAOLIN_RES_256_CACHE_ROOT')
shapenet_root = os.getenv('SHAPENET_ROOT')
cache_dir = os.path.join(cache_root, category)

categories = [category]

sdf_set = shapenet.ShapeNet_SDF_Points(root=shapenet_root, resolution=256,  categories=categories, cache_dir=cache_dir, train=True, split=1.)
point_set = shapenet.ShapeNet_Points(root=shapenet_root,  resolution=256, categories=categories, cache_dir=cache_dir, train=True, split=1.)
surface_set = shapenet.ShapeNet_Surface_Meshes(root=shapenet_root,  resolution=256, categories=categories, cache_dir=cache_dir, train=True, split=1.)
