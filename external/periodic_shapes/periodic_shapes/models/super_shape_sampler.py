import torch
from torch import nn
from periodic_shapes.layers import super_shape_functions
from periodic_shapes.models import base_shape_sampler
from periodic_shapes.models import model_utils
from periodic_shapes import utils


class SuperShapeSampler(base_shape_sampler.BaseShapeSampler):
    def __init__(self, max_m, *args, rational=True, **kwargs):
        """Intitialize SuperShapeSampler.

        Args:
            max_m: Max number of vertices in a primitive. If quadrics is True,
                then max_m must be larger than 4.
            rational: Use rational (stable gradient) version of supershapes
        """
        super().__init__(*args, **kwargs)
        self.max_m = max_m + 1
        self.rational = rational

        # Angle params
        # 1, 1, max_m, 1, 1
        index = torch.arange(start=0, end=max_m + 1,
                             step=1).view(1, 1, -1, 1, 1).float()
        # 1, 1, max_m, 1, 1
        self.angles = index / 4.
        assert [*self.angles.shape] == [1, 1, self.max_m, 1,
                                        1], (self.angles.shape, self.max_m)

    def to(self, device):
        super().to(device)
        self.angles = self.angles.to(device)

        return self

    def reshape_params(self, params):
        """
    Arguments:
      params (dict): containing supershape's parameters 
    Return:
      reshaped_primitive_params: params with additional dims for easier use in downstream process
    """
        n1 = params['n1']
        n2 = params['n2']
        n3 = params['n3']
        a = params['a']
        b = params['b']
        m_vector = params['m_vector']

        assert len(m_vector.shape) == 4
        # 1, n_primitives, max_m, 1, dim-1
        reshaped_m_vector = m_vector.view(-1, self.n_primitives, self.max_m, 1,
                                          self.dim - 1)

        # Shape params
        assert len(n1.shape) == 3
        # B=1, n_primitives, 1, 1, dim - 1
        reshaped_n1 = n1.view(-1, self.n_primitives, 1, 1, self.dim - 1)
        reshaped_n2 = n2.view(-1, self.n_primitives, 1, 1, self.dim - 1)
        reshaped_n3 = n3.view(-1, self.n_primitives, 1, 1, self.dim - 1)

        # B=1, n_primitives, 1, 1, self.dim - 1
        assert len(a.shape) == 3
        reshaped_a = a.view(-1, self.n_primitives, 1, 1, self.dim - 1)
        reshaped_b = b.view(-1, self.n_primitives, 1, 1, self.dim - 1)

        return {
            'n1': reshaped_n1,
            'n2': reshaped_n2,
            'n3': reshaped_n3,
            'a': reshaped_a,
            'b': reshaped_b,
            'm_vector': reshaped_m_vector
        }

    def get_r(self, thetas, params, *args, **kwargs):
        """Return radius.

        B, _, P, D = thetas.shape
        Args:
            thetas: (B, 1 or n_primitives, P, D)
        Returns:
            radius: (B, n_primitives, P, D)
        """
        reshaped_params = self.reshape_params(params)
        n1 = reshaped_params['n1']
        n2 = reshaped_params['n2']
        n3 = reshaped_params['n3']
        a = reshaped_params['a']
        b = reshaped_params['b']
        m_vector = reshaped_params['m_vector']

        thetas_dim_added = thetas.unsqueeze(2)
        # B, 1 or N, max_m, P, D = (1, 1, max_m, 1, 1) x (B, 1 or N, 1, P, D)
        thetas_max_m = self.angles * thetas_dim_added

        # B, 1 or N, max_m, P, D
        sin = thetas_max_m.sin()
        cos = thetas_max_m.cos()

        #r = (B, n_primitives, max_m, P, dim-1), thetas = (B, 1 or N, 1, P, dim-1)
        if self.rational:
            r = super_shape_functions.rational_supershape(
                thetas, sin, cos, n1, n2, n3, a, b)
        else:
            r = super_shape_functions.supershape(thetas, sin, cos, n1, n2, n3,
                                                 a, b)
        return (r * m_vector).sum(2)
