import sys
sys.path.append("/home/mzt/projects/yolov7/")
import torch
import torch.quantization as tq
from models.yolo import Model
import numpy as np
import cv2
import torchvision
import time

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
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]# [(label,score,x,y,w,h)]
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


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    # y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y  # [top_left_x, top_left_y, width, height]
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def evaluate(model, im0):
    from utils.datasets import letterbox
    img = letterbox(im0, 640, stride=32)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    names = model.module.names if hasattr(model, 'module') else model.names
    import random
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # device = torch.device("cuda:0")
    # model.to(device)
    # img.to(device)
    # inference
    with torch.no_grad():
        # Inference
        inference_start = time.time()
        if model.quant:
            img = tq.QuantStub()(img)
        predict = model(img, augment=False)[0]
        if model.quant:
            predict = tq.DeQuantStub()(predict)
        inference_end = time.time()

        # Apply NMS
        nms_start = time.time()
        predict = non_max_suppression(predict, conf_thres=0.5, iou_thres=0.3)
        nms_end = time.time()

    # post-processing
    results = []
    for si, pred in enumerate(predict):
        # Rescale boxes from img_size to im0 size
        from utils.general import scale_coords
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
        for *xyxy, conf, cls in pred.tolist():
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # 转xywh格式
            line = [str(int(cls)), round(conf, 2), [int(t) for t in xywh]]  # label format
            results.append(line)
            
            label = f'{names[int(cls)]} {conf:.2f}'
            from utils.plots import plot_one_box
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            cv2.imwrite('/home/mzt/projects/yolov7/demo/detect.jpg', im0)
    return results,(inference_end-inference_start, nms_end-nms_start)
    

state_dict = torch.load("/home/mzt/projects/yolov7/runs/train/45-tansuanyan_top-20221208/weights/best_state.pt", map_location="cpu")
state_dict = {k.replace("module.",""):v for k, v in state_dict.items()}
model_fp32 = Model(cfg="/home/mzt/projects/yolov7/cfg/training/yolov7.yaml", ch=3, nc=23)
model_fp32.load_state_dict(state_dict)
model_fp32.eval()

model_fp32.qconfig = tq.get_default_qconfig('fbgemm')
model_fp32_prepared = tq.prepare(model_fp32)
model_int8 = tq.convert(model_fp32_prepared)
model_int8.quant = True

for i in range(50):
    x = cv2.imread("/home/mzt/projects/yolov7/demo/images/tansuanyan copy 2.png")
    # res,(infer,nms) = evaluate(model_fp32_prepared,x)
    res,(infer,nms) = evaluate(model_int8, x)
    print(f"{i+1}"+"-"*30)
    print(f"infer\t{infer}s")
    print(f"nms\t{nms}s")
    # print(res)

