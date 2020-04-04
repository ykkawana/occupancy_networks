import trimesh
import dotenv
import os
import sys
import glob
import torch
import kaolin
import tqdm
import shutil
from concurrent import futures
from scripts import corrupted_utils

def check_and_move_corrupted_file(objfile_path):
    obj_id = None
    synset_id = None
    mesh = kaolin.rep.TriangleMesh.from_obj(objfile_path)
    try:
        v1 = torch.index_select(mesh.vertices, 0, mesh.faces[:, 0])
        v2 = torch.index_select(mesh.vertices, 0, mesh.faces[:, 1])
        v3 = torch.index_select(mesh.vertices, 0, mesh.faces[:, 2])
    except:
        print(objfile_path)
        obj_root = os.path.dirname(objfile_path)
        synset_root = os.path.dirname(obj_root)
        obj_id = os.path.basename(obj_root)
        synset_id = os.path.basename(synset_root)
        corrupted_dirpath = os.path.join(corrupted_utils.cache_root, 'corrupted', synset_id)
        if not os.path.exists(corrupted_dirpath):
            os.mkdir(corrupted_dirpath)
        shutil.move(obj_root, corrupted_dirpath)
        return obj_id, synset_id
    return obj_id, synset_id


if __name__ == '__main__':

    for synset in corrupted_utils.synset_to_label:
        label = corrupted_utils.synset_to_label[synset]
        print(label)
        root_path = os.path.join(corrupted_utils.cache_root, synset, '*/*.obj')
        objfile_paths = glob.glob(root_path)

        procs = []
        with futures.ProcessPoolExecutor(max_workers=10) as executor:
            for objfile_path in tqdm.tqdm(objfile_paths):
                proc = executor.submit(check_and_move_corrupted_file, objfile_path)
                procs.append(proc)
            results = futures.as_completed(fs=procs)

        print('done moving files, writing out...')
        with open('data/shapenetv1_corrupted_files.csv', 'w+') as f:
            for result in results:
                (synset_id, obj_id) = result.result()
                if synset_id is not None and obj_id is not None:
                    print('{},{},{}'.format(label,synset_id, obj_id), file=f)
