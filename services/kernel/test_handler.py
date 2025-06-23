import torch
from .handler import infer

def test_infer_deterministic_cpu():
    batch = torch.randn(4, 9)
    out1 = infer(batch)
    out2 = infer(batch)
    assert torch.allclose(out1, out2), "Inference is not deterministic on CPU!"
    assert out1.shape == (4, 30)

# Optionally test on GPU if available

def test_infer_deterministic_gpu():
    if not torch.cuda.is_available():
        return
    batch = torch.randn(4, 9, device='cuda')
    out1 = infer(batch.cpu()).cuda()  # Model is on CPU, so move input to CPU
    out2 = infer(batch.cpu()).cuda()
    assert torch.allclose(out1, out2), "Inference is not deterministic on GPU!"
    assert out1.shape == (4, 30) 