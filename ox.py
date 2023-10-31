import torch
import torchvision.models as models

model = models.resnet50()
sample = torch.rand(5, 3, 224, 224)

torch.onnx.export(model, sample, "resnet50.onnx", verbose=True, input_names=["input"], output_names=["output"])