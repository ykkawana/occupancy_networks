import torch
from im2mesh import eval
import time


def test_fscore():
    target_points = torch.rand([128, 2500, 3]).to('cuda')
    pred_points = torch.rand([128, 2500, 3]).to('cuda')
    ths = [0.0001, 0.01, 0.1]
    eps = 1e-7
    s = time.time()
    scores1 = eval.fscore(pred_points, target_points, ths, mode='kaolin')
    print('kaolin', time.time() - s)
    s = time.time()
    scores2 = eval.fscore(pred_points, target_points, ths, mode='pykeops')
    print('pykeops', time.time() - s)

    for key in scores1:
        t1, t2 = scores1[key], scores2[key]
        assert torch.allclose(t1, t2, atol=eps)
