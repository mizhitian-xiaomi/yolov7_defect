import os
import time

import cv2
import numpy as np
import torch
import torchvision
import yaml
from torch import nn
from tqdm import trange

from models.yolo import Model


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    # y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y  # [top_left_x, top_left_y, width, height]


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    # output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


class YoloDetector:

    def __init__(self,
                 weights,
                 yaml,
                 imgsz=640,
                 conf_thres=0.5,
                 iou_thres=0.3,
                 device="cuda:0",
                 state_dict=True):
        """
        :param weights: 模型文件
        :param yaml: 模型配置文件，包含模型结构、类别数量、类别名（顺序很重要，要和训练时顺序一致）、box阈值
        :param imgsz: 图像大小
        :param conf_thres: 置信度阈值
        :param iou_thres: iou阈值
        :param device: 选择硬件平台，默认cuda:0
        :param state_dict: true表示weights文件保存模型参数，false表示传入整个网络文件
        """

        # Initialize/load model and set device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        
        cfg_dict = self.parse_yaml(yaml)

        # bbox阈值设置
        if not cfg_dict.__contains__('thresh'):
            default_thresh = 0.6
            print(f"Warning ----------> yaml文件中请给出box阈值配置，目前默认为{default_thresh}")
            self.thresh_dict = {item:default_thresh for item in cfg_dict["names"]}
        else:
            default_thresh = float(cfg_dict["thresh"]["default_thresh"])
            if default_thresh == -1:
                self.thresh_dict = { k:float(v) for k,v in cfg_dict["thresh"].items() }
            else:
                self.thresh_dict = { k:default_thresh for k in cfg_dict["thresh"].keys() }
        
        # 加载模型权重参数的方法
        if state_dict:
            nc = cfg_dict["nc"]
            self.model = Model(yaml, ch=3, nc=nc).to(self.device)
            params = {k.replace('module.', ''): v for k, v in torch.load(weights).items()}  # 去掉module关键字，否则报错Missing
            self.model.load_state_dict(params, strict=False)
        # 加载模型网络结构的方法
        else:
            self.model = attempt_load(weights, map_location=self.device)  # load FP32 model

        # 转换标签名
        labels = cfg_dict["names"]
        self.model.names = labels
        
        self.stride = int(self.model.stride.max())  # model stride
        from utils.general import check_img_size
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        if self.half:
            self.model.half()


        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        import random
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once

        # Configure
        self.model.eval()

    def __call__(self, im0, show=False):
        """
        Args: BGR HWC image
            img:

        Returns:

        """

        if isinstance(im0, str):
            im0 = cv2.imread(im0)

        # Padded resize
        from utils.datasets import letterbox
        img = letterbox(im0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # inference
        with torch.no_grad():
            t1 = time.time()
            # Inference
            predict = self.model(img, augment=False)[0]
            t2 = time.time()
            # Apply NMS
            predict = non_max_suppression(predict, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
            t3 = time.time()
        print(f"Inference: {(t2-t1)*1E3}ms NMS: {(t3-t2)*1E3}ms ")
        
        # post-processing
        results = []
        for si, pred in enumerate(predict):
            # Rescale boxes from img_size to im0 size
            from utils.general import scale_coords
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
            for *xyxy, conf, cls in pred.tolist():
                name = self.model.names[int(cls)]
                if round(conf, 2) < self.thresh_dict[name]:
                    continue
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # 转xywh格式
                line = [name, round(conf, 2), [int(t) for t in xywh]]  # label format
                results.append(line)

                if show:
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    from utils.plots import plot_one_box
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    cv2.imshow('detect', im0)
                    cv2.waitKey(0)
        return results

    def parse_yaml(self, file):
        if file is None:
            return
        with open(file, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        return cfg_dict



if __name__ == '__main__':
    # model_path = "/run/media/cv/d/mzt/projects/yolov7/runs/train/47-density_front-20230406-train/weights/best_state.pt"
    # model_path = "/home/mzt/projects/yolov7/runs/train/tansuanyan/45-tansuanyan_top-20221208/weights/best_state.pt"
    # yaml_path = r"/home/mzt/projects/yolov7/data/tansuanyan_top.yaml"

    model_path = r"/home/mzt/work_dirs/fp16_best_state.pt"
    yaml_path = r"/home/mzt/work_dirs/tansuanyan_front_simulation.yaml"
    
    detector = YoloDetector(weights=model_path,
                            yaml=yaml_path)
    
    image_file = r"/home/mzt/projects/yolov7/demo/images/tansuanyan.png"
    bboxes = detector(image_file)
    
    T = 0
    for i in range(100):
        t0 = time.time()
        image = cv2.imread(image_file)   
        t1 = time.time()
        bboxes = detector(image)
        t2 = time.time()
        T += t2 - t1
        print(f"read: {(t1-t0)*1000}ms")
        print(f"inference+NMS:  {(t2-t1)*1000}ms")
    print(f"Average: {T/100*1000}ms")
        
    # image_dir = r"/run/media/cv/d/mzt/dataset/density/front/yolo/images"
    # for filename in os.listdir(image_dir):
    #     file_path = os.path.join(image_dir, filename)
        # predict = detector(file_path)
        # print(f" -------------------- {filename} \n{predict}\n")
