# author: mizhitian
# date: 20230518

# 测试结果
# 1920*1080输入，单卡单任务 3090Ti
# Preprocess:   5ms
# Inference:    9ms
# Postprocess:  4ms
# Sum       :   18ms

import onnx
import onnxruntime as rt
import cv2
import torch
import numpy as np
import time
import torchvision
from PIL import Image, ImageDraw, ImageFont

# hex = matplotlib.colors.TABLEAU_COLORS.values()
hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
        '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')

class Colors:
    
    """
    颜色工具类，使用颜色直接输入序号
    colors = Colors()
    color = colors(index)
    """

    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def draw_bbox(frame, bbox_list, labels=None, line_thickness=2, font_size=17, thresh=0.4):
    '''
    画bbox框
    :param frame: 视频帧 or 图像 BGR
    :param bbox_list: 检测结果 [[label, score, [xywh]]]
    :param labels: 总标签,为None时,画框颜色为FF3838
    :param line_thickness: 画框线宽
    :param font_size: 字体大小
    :param thresh: 置信度阈值
    :return: 画框后的图像
    '''
    if isinstance(frame, str):
        frame = cv2.imread(frame)
    assert isinstance(frame, np.ndarray)

    if len(bbox_list) == 0:
        return frame

    colors = Colors()
    font_path = '/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf'
    font = ImageFont.truetype(font_path, font_size)
    imc = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for bbox in bbox_list:
        score = bbox[1]
        if score < thresh:
            continue
        score = round(score, 2)
        name = bbox[0]
        color = colors(0)
        if labels and name in labels:
            index = labels.index(name)
            index = index % len(hexs) if index else 0
            color = colors(index)
        x1, y1 = bbox[2][0], bbox[2][1]
        x2, y2 = x1 + bbox[2][2], y1 + bbox[2][3]
        draw = ImageDraw.Draw(imc)
        # 画bbox框
        draw.rectangle((x1, y1, x2, y2), None, color, width=line_thickness)
        # 画标签名与置信度
        text = " {} {} ".format(name, str(score))
        fw, fh = font.getsize(text)[0], font.getsize(text)[1]  # text的宽高
        draw.rectangle((x1, y1 - fh + 1, (x1 + fw), y1),
                       fill=color, width=line_thickness)  # 标签背景
        draw.text((x1, y1 - fh + 1), text,
                  fill=(255, 255, 255), font=font)  # 标签
    # imc.show()
    return cv2.cvtColor(np.asarray(imc), cv2.COLOR_RGB2BGR)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    coords = torch.tensor(coords)
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[[0, 2]] -= pad[0]  # x padding
    coords[[1, 3]] -= pad[1]  # y padding
    coords[:4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[0].clamp_(0, img_shape[1])  # x1
    boxes[1].clamp_(0, img_shape[0])  # y1
    boxes[2].clamp_(0, img_shape[1])  # x2
    boxes[3].clamp_(0, img_shape[0])  # y2


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
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def xyxy2xywh(box):
    x1, y1, x2, y2 = box
    w = int(abs(x1 - x2))
    h = int(abs(y1 - y2))
    return [int(x1), int(y1), w, h]


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]
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
            # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            x[:, 5:] = x[:, 4:5]
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
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

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
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


class ONNXRunner():
    def __init__(self, onnx_file, class_names, device="cpu") -> None:
        """onnx模型推理类, batchsize 固定为 1

        Args:
            onnx_file (str): onnx模型文件绝对地址
            class_names (list): 分类名
            device (str, optional): 模型加载硬件平台 Defaults to "cpu".
        """

        onnx_model = onnx.load(onnx_file)  # 读取模型
        try:
            onnx.checker.check_model(onnx_model)  # 检查模型
        except Exception:
            print(f"Warning: The onnx model file <{onnx_file}> check error!!!")

        self.onnx_file = onnx_file
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.device_id = self.device.index if self.device.type != 'cpu' else -1
        self.device_id = self.device.index if self.device_id else 0 # device未指定显卡ID时默认为0号GPU
        self.half = self.device.type != 'cpu'
        self.class_names = class_names

        self.sess = None
        self.ori_shape = None
        self.resized_shape = None
        self.inputs_name = None
        self.outputs_name = None
        self.__get_sess()

        # dry run
        if self.device.type != 'cpu':
            dummy_X = torch.zeros(1, 3, *self.resized_shape[-2:],
                                  dtype=torch.float32).contiguous().numpy()
            ort_X = rt.OrtValue.ortvalue_from_numpy(
                dummy_X, self.device.type, self.device_id)

            io_binding = self.sess.io_binding()
            io_binding.bind_input(
                name=self.inputs_name,
                device_type=ort_X.device_name(),
                device_id=self.device_id,
                element_type=np.float32,
                shape=ort_X.shape(),
                buffer_ptr=ort_X.data_ptr()
            )
            io_binding.bind_output(self.outputs_name)
            self.sess.run_with_iobinding(iobinding=io_binding)

    def __call__(self, image):
        """回调函数

        Args:
            image (str, numpy.ndarray): 要检测的图像, 支持传入文件地址与numpy.ndarray格式图像

        Returns:
            list: 返回该图像中所有检测到的物体
        """
        t0 = time.time()
        if isinstance(image, str):
            image = cv2.imread(image)
        resized_img = self.preprocess(image)
        self.ori_shape = image.shape

        t1 = time.time()
        # Input on GPU, Output on CPU
        if self.device.type != "cpu":
            ort_X = rt.OrtValue.ortvalue_from_numpy(
                resized_img, self.device.type, self.device_id)
            io_binding = self.sess.io_binding()
            io_binding.bind_input(
                name=self.inputs_name,
                device_type=ort_X.device_name(),
                device_id=self.device_id,
                element_type=np.float32,
                shape=ort_X.shape(),
                buffer_ptr=ort_X.data_ptr()
            )

            # 输出在CPU,（输出在GPU需要指定shape）
            io_binding.bind_output(self.outputs_name)
            self.sess.run_with_iobinding(iobinding=io_binding)
            pred_onnx = io_binding.copy_outputs_to_cpu()[0]

        else:
            ort_X = rt.OrtValue.ortvalue_from_numpy(resized_img.numpy())
            pred_onnx = self.sess.run(
                [self.outputs_name], {self.inputs_name: ort_X})[0]
        t2 = time.time()
        result = self.postprocess(pred_onnx)
        t3 = time.time()

        print(
            f"PreProcess: {(t1-t0)*1e3}ms, Inference: {(t2-t1)*1e3}ms, PostProcess: {(t3-t2)*1e3}ms")
        return result

    def preprocess(self, img):
        """图像预处理

        Args:
            img (numpy.ndarray): 预处理的图像

        Returns:
            numpy.ndarray: 处理后的图像
        """
        img = self.__resize(img)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB CWH
        img = np.ascontiguousarray(img)  # 保证内存地址连续 Keep memory addresses contiguous
        img = torch.from_numpy(img.copy()).to(self.device) # 传输到GPU可以加速处理 Transfer to GPU for acceleration
        img = img.float()
        img = img / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img.cpu().numpy() # onnx要求numpy格式输入

    def postprocess(self, pred_onnx):
        """推理后处理: NMS、rescale bbox、调整输出bbox格式

        Args:
            pred_onnx (list): onnx 模型推理结果

        Returns:
            list: 后处理留下的所有检测框
        """
        if not isinstance(pred_onnx, torch.Tensor):
            pred_onnx = torch.tensor(pred_onnx)
        t0 = time.time()
        result = non_max_suppression(pred_onnx)[0]
        t1 = time.time()
        bboxes = []
        for _, (x1, y1, x2, y2, score, cls_id) in enumerate(result):
            label = self.class_names[int(cls_id)]
            x1, y1, x2, y2 = scale_coords(self.resized_shape[2:], [
                                          x1, y1, x2, y2], self.ori_shape[:2])
            bbox = [label, round(float(score), 2), xyxy2xywh((x1, y1, x2, y2))]
            bboxes.append(bbox)
        t2 = time.time()
        print(f"NMS: {(t1-t0)*1e3}ms, FormatResult: {(t2-t1)*1e3}ms")
        return bboxes

    def __get_sess(self):
        """获取session
        """
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type != "cpu" else [
            'CPUExecutionProvider']
        self.sess = rt.InferenceSession(self.onnx_file, providers=providers)
        self.resized_shape = self.sess.get_inputs()[0].shape
        self.inputs_name = self.sess.get_inputs()[0].name
        self.outputs_name = self.sess.get_outputs()[0].name

    def __resize(self, img, color=(114, 114, 114)):
        """输入图像resize到指定大小, 不改变图像比例, 缺失部分填充(114,114,114)

        Args:
            img (ndarray): 原始输入图像
            color (tuple, optional): padding填充的颜色. Defaults to (114, 114, 114).

        Returns:
            ndarray: 返回resize后的新图像
        """
        target_shape = self.resized_shape[2:]
        shape = img.shape[:2]

        r = min(target_shape[0]/shape[0], target_shape[1]/shape[1])
        target_unpad = int(round(shape[1]*r)), int(round(shape[0]*r))
        dw, dh = target_shape[1] - \
            target_unpad[0], target_shape[0] - target_unpad[1]

        img = cv2.resize(img, target_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)/2), int(round(dh + 0.1)/2)
        left, right = int(round(dw - 0.1)/2), int(round(dw + 0.1)/2)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img

    @staticmethod
    def draw(image, bboxes, class_names):
        """静态方法, 给图像画框

        Args:
            image (str或者numpy.ndarray): 需要画框的图像
            bboxes (list): 检测框信息，[label, score, [x1, y1, x2, y2]]
            class_names (list): 类别名

        Returns:
            numpy.ndarray: 返回画好框的图像
        """
        image = draw_bbox(frame=image, bbox_list=bboxes, labels=class_names)
        return image


if __name__ == "__main__":
    names = ['tube_plug',
             'tube_noplug',
             'tweezer',
             'scale_pad',
             'scale_nopad',
             'narrow_bottle_cap',
             'wide_bottle_cap',
             'bottle_nocap',
             'bottle_cap_on',
             'bottle_cap_down',
             'dropper',
             'red_dropping_bottle',
             'white_dropping_bottle',
             'red_line',
             'blue_line',
             '90_pipe',
             'gas_guide_tube',
             'color_ring',
             'iron_supporting',
             'tube_clamp',
             'tube_hand',
             '250_beaker',
             'green_line'
             ]
    onnx_file = r"/home/mzt/projects/yolov7/runs/train/tansuanyan/45-tansuanyan_top-20221208/weights/best-test-simplify-grid.onnx"
    image_file = r"/home/mzt/projects/yolov7/demo/images/tansuanyan.png"

    names = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
         '11', '12', '13', '14', '15', '16', '17', '18', '19' ]
    onnx_file = r"/home/mzt/work_dirs/dianxue/best-simplify-fp16.onnx"
    image_file = r"/home/mzt/projects/yolov7/demo/images/dianxue.png"
    # 使用 onnxrunner
    run = ONNXRunner(onnx_file, class_names=names, device="cuda:0")
    T = 0
    for i in range(100):
        img = cv2.imread(image_file)
        t0 = time.time()
        bboxes = run(img) # recall
        t1 = time.time()
        T += (t1-t0)
        
        print(f"{i} ----------------------------")
        print(f"sum = {(t1-t0)*1e3}ms")
        print(bboxes)
    image = ONNXRunner.draw(image_file, bboxes, run.class_names)
    cv2.imwrite(
        f"./demo/images/dianxue-detected.png", image)
    print(f"Average = {T*1e3/1e2}ms")

