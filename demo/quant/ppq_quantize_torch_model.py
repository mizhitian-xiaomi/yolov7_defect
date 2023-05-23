import sys

sys.path.append("/home/mzt/projects/yolov7")
from typing import Iterable

import torch
import torchvision
from torch.utils.data import DataLoader

from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_torch_model

from models.experimental import attempt_load
from models.yolo import Model

BATCHSIZE = 1
INPUT_SHAPE = [3, 640, 640]
DEVICE = 'cuda' # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.PPL_CUDA_INT8  # identify a target platform for your network.

def load_calibration_dataset() -> Iterable:
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

# Load a pretrained mobilenet v2 model
# model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
# checkpoint = r'/home/mzt/projects/yolov7/runs/train/45-evaporative_crystallization_top-20230331-train/weights/best_state.pt'
# yaml = r'/home/mzt/projects/yolov7/cfg/training/yolov7.yaml'
# nc = 18
# model = Model(yaml, ch=3, nc=nc).to(DEVICE)
# params = {k.replace('module.', ''): v for k, v in torch.load(checkpoint, map_location=DEVICE).items()}  # 去掉module关键字，否则报错Missing
# model.load_state_dict(params)
checkpoint = r'/home/mzt/projects/yolov7/runs/train/45-evaporative_crystallization_top-20230331-train/weights/best.pt'
model = attempt_load(checkpoint, map_location=DEVICE)
model = model.to(DEVICE)

# create a setting for quantizing your network with PPL CUDA.
quant_setting = QuantizationSettingFactory.pplcuda_setting()
quant_setting.equalization = True # use layerwise equalization algorithm.
quant_setting.dispatcher   = 'conservative' # dispatch this network in conservertive way.

# Load training data for creating a calibration dataloader.
calibration_dataset = load_calibration_dataset()
calibration_dataloader = DataLoader(
    dataset=calibration_dataset,
    batch_size=BATCHSIZE, shuffle=True)

# quantize your model.
quantized = quantize_torch_model(
    model=model, calib_dataloader=calibration_dataloader,
    calib_steps=32, input_shape=[BATCHSIZE] + INPUT_SHAPE,
    setting=quant_setting, collate_fn=collate_fn, platform=PLATFORM,
    onnx_export_file='/home/mzt/projects/yolov7/quant/output/model.onnx', device=DEVICE, verbose=0)

# Quantization Result is a PPQ BaseGraph instance.
assert isinstance(quantized, BaseGraph)

# export quantized graph.
export_ppq_graph(graph=quantized, platform=PLATFORM,
                 graph_save_to='/home/mzt/projects/yolov7/quant/output/quantized(onnx).onnx',
                 config_save_to='/home/mzt/projects/yolov7/quant/output/quantized(onnx).json')
