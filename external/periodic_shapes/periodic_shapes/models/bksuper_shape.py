import torch
from torch import nn
import math

from models import model_utils


class SuperShapes(nn.Module):
    def __init__(self,
                 max_m,
                 n_primitives,
                 rational=True,
                 train_logits=True,
                 train_linear_scale=True,
                 quadrics=False,
                 train_ab=True,
                 transition_range=1.,
                 dim=2):
        """Initialize SuperShapes.

        Args:
            max_m: Max number of vertices in a primitive. If quadrics is True,
                then max_m must be larger than 4.
            n_primitives: Number of primitives to use.
            rational: Use rational (stable gradient) version of supershapes
            debug: debug mode.
            train_logits: train logits flag.
            train_linear_scale: train linear scale flag.
            quadrics: quadrics mode flag. In this mode, n1, n2, n3 are set
                to one and not trained, and logits is not trained and 5th 
                element of m_vector is always one and the rest are zero.
            dim: if dim == 2, then 2D mode. If 3, 3D mode.
        Raises:
            NotImplementedError: when dim is neither 2 nor 3.
        """
        super().__init__()
        self.n_primitives = n_primitives
        self.max_m = max_m + 1
        self.rational = rational
        self.quadrics = quadrics
        self.train_logits = train_logits
        self.train_ab = train_ab
        self.dim = dim
        self.transition_range = transition_range
        if not self.dim in [2, 3]:
            raise NotImplementedError('dim must be either 2 or 3.')
        if self.rational:
            self.n23scale = 10.
            self.n23bias = 1.
        else:
            self.n23scale = 1.
            self.n23bias = 0.

        if self.quadrics:
            self.n23scale = 1.
            self.n23bias = 1e-7

        if self.quadrics:
            assert max_m >= 4, 'super quadrics musth have m bigger than 4.'

        # 1, n_primitives, max_m, 1, dim-1
        logits_list = []
        for idx in range(self.dim - 1):
            if self.quadrics:
                logits = torch.eye(self.max_m).view(
                    1, 1, self.max_m, self.max_m)[:, :, 4, :].repeat(
                        1, n_primitives, 1).float() * 10
            elif not self.train_logits:
                logits = torch.eye(self.max_m).view(
                    1, 1, self.max_m,
                    self.max_m)[:, :, (self.max_m - 1), :].repeat(
                        1, n_primitives, 1).float() * 10
            else:
                logits = torch.Tensor(1, n_primitives, self.max_m)
            logits_list.append(logits)
        self.logits = torch.stack(logits_list, axis=-1)
        assert [*self.logits.shape
                ] == [1, n_primitives, self.max_m, self.dim - 1]
        if not self.quadrics and self.train_logits:
            self.logits = nn.Parameter(self.logits)

        # Shape params
        # B=1, n_primitives, 1, 1
        self.n1 = nn.Parameter(torch.Tensor(1, self.n_primitives,
                                            self.dim - 1))
        self.n2 = nn.Parameter(torch.Tensor(1, self.n_primitives,
                                            self.dim - 1))
        self.n3 = nn.Parameter(torch.Tensor(1, self.n_primitives,
                                            self.dim - 1))

        assert [*self.n1.shape] == [1, self.n_primitives, self.dim - 1]
        # Pose params
        # B=1, n_primitives, 2
        self.transition = nn.Parameter(torch.Tensor(1, n_primitives, self.dim))
        # B=1, n_primitives, 1
        if self.dim == 2:
            self.rotation = nn.Parameter(torch.Tensor(1, n_primitives, 1))
        else:
            # Quaternion
            self.rotation = nn.Parameter(torch.Tensor(1, n_primitives, 4))

        # B=1, n_primitives, 2
        self.linear_scale = nn.Parameter(
            torch.Tensor(1, n_primitives, self.dim))
        self.linear_scale.requires_grad = train_linear_scale

        # B=1, n_primitives, 1, 1
        self.a = nn.Parameter(torch.Tensor(1, self.n_primitives, self.dim - 1))
        self.b = nn.Parameter(torch.Tensor(1, self.n_primitives, self.dim - 1))
        assert [*self.a.shape] == [1, self.n_primitives, self.dim - 1]
        if not self.train_ab:
            self.a.requires_grad = False
            self.b.requires_grad = False

        # B=1, n_primitives
        self.prob = nn.Parameter(torch.Tensor(1, n_primitives))

        if self.quadrics:
            self.n1.requires_grad = False
            self.logits.requires_grad = False
        elif not self.train_logits:
            self.logits.requires_grad = False

        self.weight_init()

    def weight_init(self):
        if self.quadrics:
            torch.nn.init.ones_(self.n1)
            torch.nn.init.uniform_(self.n2, 0, 1)
            torch.nn.init.uniform_(self.n3, 0, 1)
        elif not self.train_logits:
            torch.nn.init.uniform_(self.n1, 0, 1)
            torch.nn.init.uniform_(self.n2, 0, 1)
            torch.nn.init.uniform_(self.n3, 0, 1)
        else:
            torch.nn.init.uniform_(self.n1, 0, 1)
            torch.nn.init.uniform_(self.n2, 0, 1)
            torch.nn.init.uniform_(self.n3, 0, 1)
            torch.nn.init.uniform_(self.logits, 0, 1)
        if self.train_ab:
            torch.nn.init.uniform_(self.a, 0.9, 1.1)
            torch.nn.init.uniform_(self.b, 0.9, 1.1)
        else:
            torch.nn.init.ones_(self.a)
            torch.nn.init.ones_(self.b)
        torch.nn.init.uniform_(self.rotation, 0, 1)
        torch.nn.init.uniform_(self.linear_scale, 0, 1)
        torch.nn.init.uniform_(self.transition, -self.transition_range, self.transition_range)
        torch.nn.init.uniform_(self.prob, 0, 1)

    def get_primitive_params(self, x):
        B = x.shape[0]
        probabilistic = self.training and self.train_logits and not self.quadrics
        m_vector = model_utils.get_m_vector(self.logits,
                                            probabilistic=probabilistic)
        m_vector = m_vector.repeat(B, 1, 1, 1, 1)

        linear_scale = nn.functional.tanh(self.linear_scale) + 1.1
        linear_scale = linear_scale.repeat(B, 1, 1)

        a = nn.functional.relu(self.a)
        a = a.repeat(B, 1, 1, 1)
        b = nn.functional.relu(self.b)
        b = b.repeat(B, 1, 1, 1)

        n1 = nn.functional.relu(self.n1)
        n1 = n1.repeat(B, 1, 1, 1)
        n2 = nn.functional.relu(self.n2 * self.n23scale) + self.n23bias
        n2 = n2.repeat(B, 1, 1, 1)
        n3 = nn.functional.relu(self.n3 * self.n23scale) + self.n23bias
        n3 = n3.repeat(B, 1, 1, 1)

        #rotation = nn.functional.tanh(self.rotation) * math.pi
        rotation = nn.functional.normalize(self.rotation, dim=-1)
        rotation = rotation.repeat(B, 1, 1)
        
        transition = self.transition.repeat(B, 1, 1)

        linear_scale = linear_scale.repeat(B, 1, 1)

        prob = nn.functional.sigmoid(self.prob)
        prob = prob.repeat(B, 1)

        return {
            'n1': n1,
            'n2': n2,
            'n3': n3,
            'a': a,
            'b': b,
            'm_vector': m_vector,
            'rotation': rotation,
            'transition': transition,
            'linear_scale': linear_scale,
            'prob': prob
        }

    def forward(self, x):
        return self.get_primitive_params(x)

    def to(self, device):
        super().to(device)
        if self.quadrics or not self.train_logits:
            self.logits = self.logits.to(device)

        return self
