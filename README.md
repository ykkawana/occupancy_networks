# PSNet experiments

## Directory structure
`external/<model of previous works>/`: Place model of external works. It needs to be modularized.
`out/submission/eval/`: Evaluation results are stored.
`paper_resources/`: Tex resources and their generation scripts are stored.

## Prerequiste
### venv
```
source env/bin/activate
```
### PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:external/periodic_shapes:$PWD:external/atlasnetv2
```

## Training
### Single GPU
```
CUDA_VISIBLE_DEVICES=0 python3 train.py out/img/pnet.config
```
### Multiple GPUs
```
CUDA_VISIBLE_DEVICES=0,1,2 python3 dist_train.py out/img/pnet.config
```

## Evaluation
### IoU
```
CUDA_VISIBLE_DEVICES=0 python3 eval.py out/img/pnet.yamla <config overwrite options. e.g. --test.threshold 0.999999>
```
This creates copied run directory `out/submission/eval/img/<out_dir>_<YYYYMMDD>`.
`<out_dir>` is written in config yaml.
For PSNet, `--test.threshold 0.999999` is usually used for IoU.
#### Generate IoU result in existing directory
Use `--dontcopy` option with config.yaml created by eval.py.
E.g.
```
CUDA_VISIBLE_DEVICES=0 python3 eval.py out/submission/eval/img/config.yaml
```
#### Generate IoU result of the run which doesn't have output dir under out/.
Use `--no_copy_but_create_new` option.

#### Generate IoU result for BSPNet
```
sh external/bspnet/scripts/eval_iou.sh <config path> <GPU id>
```

### Generate Mesh
For PSNet and AtlasNet V2,
```
sh im2mesh/<pnet or atlasnetv2>/script/generate_mesh.sh out/submission/eval/img/<out_dir>_<YYYYMMDD>/config.yaml
```
This automatically creates config yaml:
```
im2mesh/<pnet or atlasnetv2>/script/generate_mesh.sh out/submission/eval/img/<out_dir>_<YYYYMMDD>/gen_***_<yyyymmdd>.yaml <GPU id>
```

### Fscore
Use generated config file by mesh generation.
```
CUDA_VISIBLE_DEVICES=0 python3 eval_fscore.py out/submission/eval/img/<out_dir>_<YYYYMMDD>/gen_***_<yyyymmdd>.yaml
```

### Chamfer distance, normal consistency
Use generated config file by mesh generation.
```
CUDA_VISIBLE_DEVICES=0 python3 eval_mesh.py out/submission/eval/img/<out_dir>_<YYYYMMDD>/gen_***_<yyyymmdd>.yaml
```

## Experiment specific
All paper related scripts are in `paper_resources/`.
Generated resources for tex are stored in `paper_resources/<experiment>/resoureces`.
Some experiments requires to run IoU and fscore evaluation before the experiment.
Scripts for generating resources are stored in `paper_resources/<experiment>/scripts`.
### Mesh quality comparison
Under `paper_resources/compare_mesh_methods`.
Run `paper_resources/compare_mesh_methods/scripts/mesh_quality_comparison.ipynb`.

### SVR Pix3D evaluation
Under `paper_resources/pix3d_comparison`
Fist, You need to create OccNet experiment environment compatible Pix3D dataset.
Run scripts in following steps to create the Dataset.
1. Compute stats of ShapeNet renderings margin.
```
python3 paper_resources/pix3d_comparison/scripts/calc_stats_of_margin_of_rendering.py
```
2. Create the compatible renderings of Pix3D.
```
python3 paper_resources/pix3d_comparison/scripts/convert_pix3d_images.py
```
3. Create the surface mesh.
```
python3 paper_resources/pix3d_comparison/scripts/convert_pix3d_models.py
```
4. Create sample lists.
```
python3 paper_resources/pix3d_comparison/scripts/generate_list.py
```
After finising Pix3D dataset conversion, generate resoureces.
1. Generate mesh for Pix3D for PSNet (for atlasnetv2, TBD)
```
sh paper_resources/pix3d_comparison/scripts/generate_explicit_pix3d_mesh.sh  out/submission/eval/img/<out_dir>_<YYYYMMDD>/config.yaml <GPU id>
```
3. Generate Fscore
2. Generate resoureces
```
CUDA_VISIBLE_DEVICES=0 python3 paper_resources/pix3d_comparison/scripts/generate_pix3d_table_and_generate_renderings.py
```
### SVR ShapeNet evaluation.
Under `paper_resources/shapenet_svr_comparison`.
1. Generate IoU, fscore, CD1 metrics.
2. Prep csv tables of metrics of previous works under `paper_resources/shapenet_svr_comparison/csv`.
3. Run `paper_resources/shapenet_svr_comparison/scripts/generate_svr_table.ipynb`.
4. Tex table is generated in `paper_resources/shapenet_svr_comparison/table.txt`.

### Primitive visualization
This experiment render primitives and evaluate part semseg.
For visualize primitives,
```
CUDA_VISIBLE_DEVICES=0 python3 paper_resources/primitive_visualization/scripts/render_primitives.py
```
For evaluate part semseg,
1. Generate semseg data.
```
python3 paper_resources/primitive_visualization/scripts/generate_semseg_data.py
``` 
2. TBD

## Other notes
### Rendering
Rendering script is in `/home/mil/kawana/workspace/RenderForCNN`.
Wrapper for this project is in `scripts/render_3dobj.sh`.

### Visualization
Simple mesh visualization by trimesh, see `scripts/show_mesh.py`.
Point cloud visualization by plotly, see `external/periodic_shapes/periodic_shapes/visualize/plot.py`.
