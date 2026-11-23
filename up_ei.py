import os
import torch
import onnx
from model import ColorAudioCNN

model = ColorAudioCNN(num_classes=3)
model.load_state_dict(torch.load("color_model.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 64, 63)
onnx_file = "color_voice_lock.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_file,
    export_params=True,
    opset_version=13,
    dynamo=False,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

onnx_model = onnx.load(onnx_file, load_external_data=True)
onnx.save_model(onnx_model, onnx_file, save_as_external_data=False)

data_file = onnx_file + ".data"
if os.path.exists(data_file):
    os.remove(data_file)

print(f"SUCCESS! Single self-contained file ready: {onnx_file} ({os.path.getsize(onnx_file) / 1024:.1f} KB)")
