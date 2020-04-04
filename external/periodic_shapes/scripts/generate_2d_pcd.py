import os, sys, glob
import random
import skimage.io
import skimage.filters
import numpy as np
import pickle

random.seed(0)
filepaths = [path.strip() for path in open('02691156.txt').readlines()]
chosen = random.sample(filepaths, 3)
print(chosen)
margin = 5 
for idx, path in enumerate(chosen):
    print(path)
    img = skimage.io.imread(path)
    print(img.shape)
    mask = img[:, :, 3]
    shape= mask.shape[:2]

    coordx = np.where(mask == mask.max())[1].astype(np.float32)
    coordy = np.where(mask == mask.max())[0].astype(np.float32)
    minx = coordx.min().astype(np.uint8)
    miny = coordy.min().astype(np.uint8)
    maxx = coordx.max().astype(np.uint8)
    maxy = coordy.max().astype(np.uint8)

    mask = mask[miny-margin:maxy+margin, minx-margin:maxx+margin]

    mask = skimage.transform.resize(mask, (300, 300))
    mask = mask.astype(np.uint8).astype(np.float32)/mask.max()
    edge = skimage.filters.laplace(mask, ksize=3)
    edge2 = np.copy(edge)
    edge2[edge>0.5] = 1
    edge2[edge<=0.5] = 0
    edge2*=255
    skimage.io.imsave('{}.png'.format(idx), edge2.astype(np.uint8))
    coordx = np.where(edge2 == 255)[1].astype(np.float32)
    coordy = np.where(edge2 == 255)[0].astype(np.float32)
    coordxsave = (coordx - coordx.min()) / (coordx.max()-coordx.min()) - 0.5
    coordysave = (coordy - coordy.min()) / (coordy.max()-coordy.min()) - 0.5
    coordxsave *= 5.
    coordysave *= 5.
    coord = [coordxsave, coordysave]
    pickle.dump(coord, open('{}.pkl'.format(idx), 'wb'))

    std = np.zeros_like(mask).astype(np.float32) - 1.
    std[mask == 1] = 1.
    std[edge2==255] = 0.
    assert std[coordy[0].astype(np.uint8), coordx[0].astype(np.uint8)] == 0
    rangex, rangey = np.meshgrid(range(0, 300), range(0, 300)) 
    rangecoordx = (rangex.reshape(-1) - coordx.min()) / (coordx.max()-coordx.min()) - 0.5
    rangecoordy = (rangey.reshape(-1) - coordy.min()) / (coordy.max()-coordy.min()) - 0.5
    rangecoordx *= 5.
    rangecoordy *= 5.
    rangecoord = [rangecoordx, rangecoordy, std.reshape(-1)]

    print(rangecoordx.shape)
    pickle.dump(rangecoord, open('{}_std.pkl'.format(idx), 'wb'))
    print(rangecoordx.max(), rangecoordx.min())


