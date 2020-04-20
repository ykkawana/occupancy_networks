from periodic_shapes.layers import super_shape_functions
import torch
import math

angles = (torch.rand([3, 5, 2]) * math.pi).clamp(max=math.pi, min=0)
radius = torch.ones([3, 5, 2])
coord = super_shape_functions.sphere2cartesian(radius, angles)
radius2, angles2 = super_shape_functions.cartesian2sphere(coord)
print(radius2.shape, angles2.shape)
coord2 = super_shape_functions.sphere2cartesian(radius2, angles2)
print(coord)
print(coord2)
assert torch.allclose(coord, coord2), (coord - coord2)
assert torch.all(torch.eq(radius.mean(), radius2.mean()))
assert torch.allclose(
    angles[..., 0].sin(),
    angles2[...,
            0].sin()), (angles.sin().mean(), angles2.sin().mean(),
                        angles2.mean() * 3, (angles - angles2) / math.pi * 180)
