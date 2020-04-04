from models import super_shape
import utils
import torch
from unittest import mock
import math

m = 2
n1 = 1
n2 = 1
n3 = 1
a = 1
b = 1
theta = math.pi / 2.


@mock.patch('models.model_utils.get_m_vector')
def test_get_primitive_params_m_vector_probabilistic(mocker):
    """Test if m vector is returned probabilistically or not.
    It's probabilistic in quadrics mode is false, eval mode, and train logits options is false. 
    """

    (theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor, n3_tensor,
     ainv_tensor,
     binv_tensor) = utils.get_single_input_element(theta, m, n1, n2, n3, a, b)

    p = super_shape.SuperShapes(m, 1, use_paramnet=False)
    p.eval()

    x = torch.zeros([1, 1]).float()
    p.get_primitive_params(x)

    args, kwargs = mocker.call_args
    logits = args[0]
    assert torch.all(torch.eq(logits, p.logits))
    assert not kwargs['probabilistic']

    p.train()
    p.get_primitive_params(x)

    args, kwargs = mocker.call_args
    logits = args[0]
    assert torch.all(torch.eq(logits, p.logits))
    assert kwargs['probabilistic']

    p_quadrics = super_shape.SuperShapes(4, 1, quadrics=True, train_logits=False, use_paramnet=False)

    p_quadrics.get_primitive_params(x)
    args, kwargs = mocker.call_args
    logits = args[0]
    assert torch.all(torch.eq(logits, p_quadrics.logits))
    assert not kwargs['probabilistic']

    p_quadrics.train()
    p_quadrics.get_primitive_params(x)
    args, kwargs = mocker.call_args
    logits = args[0]
    assert torch.all(torch.eq(logits, p_quadrics.logits))
    assert not kwargs['probabilistic']

def test_get_primitive_params_with_net():
    p_quadrics = super_shape.SuperShapes(4, 1, quadrics=True, train_logits=False, use_paramnet=True, latent_dim=1)
    x = torch.zeros([1, 1]).float()
    m_vector = p_quadrics.get_primitive_params(x)['m_vector']
    n1 = p_quadrics.get_primitive_params(x)['n1']
    target_n1 = torch.tensor([[[1.]]])
    target_m_vector = torch.tensor([[[[0.],
          [0.],
          [0.],
          [0.],
          [1.]]]])
    assert torch.all(torch.eq(target_m_vector, m_vector))
    assert torch.all(torch.eq(target_n1, n1))

    p_quadrics = super_shape.SuperShapes(4, 1, quadrics=False, train_logits=True, use_paramnet=True, latent_dim=1)
    m_vector = p_quadrics.get_primitive_params(x)['m_vector']
    n1 = p_quadrics.get_primitive_params(x)['n1']
    assert not torch.all(torch.eq(target_m_vector, m_vector))
    assert not torch.all(torch.eq(target_n1, n1))


def test_init():
    """Test tensor shapes."""
    # 2D
    super_shape.SuperShapes(m, 1, dim=2)
    # 3D
    super_shape.SuperShapes(m, 1, dim=3)
