import torch
import os
from nn_model import PredictionModel
import copy

MODEL_PATH = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/model_Test Loss: 0.8945_2024-10-09 20:35:59_sat.pth"
EXPORT_PATH = os.path.join(os.path.dirname(__file__), "models/ocean_tensorscript.pt")

if not os.path.exists(os.path.dirname(EXPORT_PATH)):
    os.makedirs(os.path.dirname(EXPORT_PATH))

def main():
    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = PredictionModel(input_dim=9, layers_config=[512, 512], output_dim=30, dropout_prob=0.2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Example input for tracing (batch size 1, input_dim=9)
    example = torch.randn(1, 9)
    traced = torch.jit.trace(model, example)
    traced.save(EXPORT_PATH)
    print(f"TorchScript model exported to {EXPORT_PATH}")

if __name__ == "__main__":
    main() 