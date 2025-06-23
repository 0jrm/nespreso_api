import torch
import os
import copy

_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/ocean_tensorscript.pt')
_CHECKPOINT_PATH = '/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/model_Test Loss: 0.8945_2024-10-09 20:35:59_sat.pth'
_model = None
_pca_temp = None
_pca_sal = None
_input_params = None


def _get_model():
    global _model
    if _model is None:
        _model = torch.jit.load(_MODEL_PATH, map_location='cpu')
        _model.eval()
    return _model

def infer(batch: torch.Tensor) -> torch.Tensor:
    """
    Run inference on the given batch tensor using the TorchScript model.
    Args:
        batch (torch.Tensor): Input tensor of shape (N, 9)
    Returns:
        torch.Tensor: Output tensor of shape (N, 30)
    """
    model = _get_model()
    with torch.no_grad():
        return model(batch)

def get_pca_objects():
    global _pca_temp, _pca_sal, _input_params
    if _pca_temp is None or _pca_sal is None or _input_params is None:
        checkpoint = torch.load(_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        _pca_temp = checkpoint['pca_temp']
        _pca_sal = checkpoint['pca_sal']
        _input_params = checkpoint['input_params']
    return _pca_temp, _pca_sal, _input_params 