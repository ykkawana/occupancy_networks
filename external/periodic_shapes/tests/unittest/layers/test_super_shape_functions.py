import torch
from layers import super_shape_functions
import math


def test_rational_super_shape():
    m = 4
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1

    theta = torch.tensor([math.pi / 2])
    mtheta = m * theta / 4
    r = super_shape_functions.rational_supershape(theta, mtheta.sin(),
                                                  mtheta.cos(), n1, n2, n3, a,
                                                  b)

    assert r.item() == 1.
