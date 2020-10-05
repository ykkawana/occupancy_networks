# %%
import pandas as pd
import numpy as np

# %%
path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization/resources/primitive_complexity/gaussian_curvature_even_faces_001.pkl'
df = pd.read_pickle(path)

# %%
df['abs_trimesh_mean_curv'] = df['vertices'] * df['abs_trimesh_mean_curv_mean']
df['abs_trimesh_gauss_curv'] = df['vertices'] * df[
    'abs_trimesh_gauss_curv_mean']


#%%
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average)**2, weights=weights)
    return (average, np.sqrt(variance))


# %%
models = ['bsp', 'sh', 'sq']
metrics = [
    'abs_trimesh_gauss_curv_mean', 'abs_trimesh_mean_curv_mean',
    'abs_trimesh_gauss_curv_std'
]
for metric in metrics:
    for model in models:
        values = df[df['model_name'] == model][metric]
        weights = df[df['model_name'] == model]['vertices']
        weights = weights / weights.sum()
        print(model, metric, weighted_avg_and_std(values, weights))
