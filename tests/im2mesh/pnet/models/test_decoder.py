from im2mesh.pnet.models import decoder
import torch


def test_to_device():
    net = decoder.PeriodicShapeDecoderSimplest(is_quadrics=True)
    device = torch.device('cuda:0')
    net.to(device)
    assert net.primitive.abn1n2n3_mask.device == device


def test_eval():
    net = decoder.PeriodicShapeDecoderSimplest(is_quadrics=True)
    net.train()
    assert net.training
    assert net.primitive.training
    net.eval()
    assert not net.training
    assert not net.primitive.training
