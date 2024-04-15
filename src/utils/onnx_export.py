import torch
from pathlib import Path
import torch._C._onnx as _C_onnx


def save_model_onnx(model, input_values, input_names, output_names, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        input_values,
        Path(directory) / "model.onnx",
        opset_version=16,
        input_names=input_names,
        output_names=output_names,
        export_params=False,
        operator_export_type=_C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
