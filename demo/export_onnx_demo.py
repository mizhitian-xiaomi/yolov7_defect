import os
import math
import torch
import torch.nn as nn
import onnx
import onnxsim
from onnxmltools.utils import float16_converter
import sys
sys.path.append("./")
import models

# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------


class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    # ceil gs-multiple 得到可被s整除的img_size值
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' %
              (img_size, s, new_size))
    return new_size


class ExportONNXUtils():
    def __init__(self) -> None:
        pass

    @staticmethod
    def export_for_yolov7(checkpoint_path, img_size=[640, 640], batch_size=1, device="cuda:0"):
        export_file = os.path.join(
            os.path.dirname(checkpoint_path), os.path.basename(checkpoint_path).replace(".pt", ".onnx"))  # filename
        ckpt = torch.load(checkpoint_path, map_location=device)  # load
        model = ckpt['ema' if ckpt.get(
            'ema') else 'model'].float().fuse().eval()  # FP32 model
        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        # Checks
        gs = int(max(model.stride))  # grid size (max stride)
        # verify img_size are gs-multiples
        img_size = [check_img_size(x, gs) for x in img_size]
        # Update model
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            # elif isinstance(m, models.yolo.Detect):
            #     m.forward = m.forward_export  # assign forward (optional)
        model.model[-1].export = False  # set Detect() layer grid export
        # image size(1,3,320,192) iDetection
        dummy_input = torch.zeros(batch_size, 3, *img_size).to(device)
        y = model(dummy_input)  # dry run

        # ONNX export
        try:
            print('\n----- Starting ONNX export with onnx %s...' %
                  onnx.__version__)
            model.eval()
            output_names = ['output']
            dynamic_axes = None
            model.model[-1].concat = True
            torch.onnx.export(model, dummy_input, export_file,
                              verbose=False,
                              opset_version=13,
                              input_names=['images'],
                              output_names=output_names,
                              dynamic_axes=dynamic_axes)
            # Checks
            onnx_model = onnx.load(export_file)
            onnx.checker.check_model(onnx_model)  # check onnx model
            onnx.save(onnx_model, export_file)
            print(f'----- Export ONNX save: {export_file}.')

            # Simplify
            onnx_model = ExportONNXUtils.simplify(onnx_model)
            simplify_file = export_file.replace(".onnx", "-simplify.onnx")
            onnx.save(onnx_model, simplify_file)
            print(f'----- Simplify ONNX save: {simplify_file}.')

            # fp32 转 fp16
            trans_model = ExportONNXUtils.convert2fp16(onnx_model)
            trans_file = simplify_file.replace(".onnx", "-fp16.onnx")
            onnx.save_model(trans_model, trans_file)
            print('----- ONNX save simplify-fp16 model: %s' % trans_file)

        except Exception as e:
            print('----- ONNX export failure: %s' % e)

    def simplify(onnx_model):

        if isinstance(onnx_model, str) and onnx_model.endswith(".onnx") and os.path.isfile(onnx_model):
            onnx_model = onnx.load(onnx_model)

        try:
            print('\n----- Starting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            if not check:
                print(f"Warning: ONNX simplify check error: \n{e}")
        except Exception as e:
            print(f'----- Simplify ONNX failure: \n{e}')

        return onnx_model

    def convert2fp16(onnx_model):
        """onnx model转fp16精度

        Args:
            onnx_model (onnx model): onnx模型

        Returns:
            trans_model (onnx model): 转换后的onnx模型
        """

        if isinstance(onnx_model, str) and onnx_model.endswith(".onnx") and os.path.isfile(onnx_model):
            onnx_model = onnx.load(onnx_model)

        try:
            print('\n----- Starting to convert ONNX to FP16...')
            trans_model = float16_converter.convert_float_to_float16(
                onnx_model, keep_io_types=True)  # 转fp16
            print("----- ONNX convert to FP16 success.")
            try:
                onnx.checker.check_model(trans_model)  # 检查模型，报错但不影响模型使用
            except Exception as e:
                print(f"Warning: ONNX convert to FP16 check error: \n{e}")
        except Exception as e2:
            print(f"Error: ONNX convert to FP16 faild: \n{e2}")
        return trans_model


if __name__ == "__main__":
    checkpoint_path = r"/home/mzt/work_dirs/dianxue/best.pt"
    img_size = [640, 640]
    batch_size = 1
    device = "cuda:0"
    ExportONNXUtils.export_for_yolov7(checkpoint_path)
