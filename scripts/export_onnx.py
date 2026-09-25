"""Exports the MiniFASNet .pth models to ONNX (next to the originals).

Needs the dev dependencies (requirements-dev.txt); torch is NOT needed at runtime.
Usage: python scripts/export_onnx.py
"""

import sys
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anti_spoof.model_lib.MiniFASNet import (  # noqa: E402
    MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE,
)
from anti_spoof.predictor import MODELS_DIR  # noqa: E402
from anti_spoof.utils import get_kernel, parse_model_name  # noqa: E402

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE,
}


def load_model(model_path):
    h_input, w_input, model_type, _ = parse_model_name(model_path.name)
    model = MODEL_MAPPING[model_type](conv6_kernel=get_kernel(h_input, w_input))

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if next(iter(state_dict)).startswith('module.'):
        state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
    model.load_state_dict(state_dict)
    model.eval()
    return model, h_input, w_input


def main():
    for pth_path in sorted(MODELS_DIR.glob("*.pth")):
        model, h_input, w_input = load_model(pth_path)
        onnx_path = pth_path.with_suffix(".onnx")
        torch.onnx.export(
            model,
            (torch.zeros(1, 3, h_input, w_input),),
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            external_data=False,
            verbose=False,
        )
        print(f"{pth_path.name} -> {onnx_path.name}")


if __name__ == "__main__":
    main()
