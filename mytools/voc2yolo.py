import xml.etree.ElementTree as ET
import os.path as osp
import os
from os import getcwd

from tqdm import tqdm

sets = ['train', 'test', 'val']

base_dir = r"/run/media/cv/d/mzt/dataset/tansuanyan/top"
voc_ann = os.path.join(base_dir, "VOC2007","Annotations")
voc_txt = os.path.join(base_dir, "VOC2007","ImageSets", "Main")


# 模拟版碳酸盐
# top
# classes =  [
#   'tube_plug',
#   'tube_noplug',
#   'tweezer',
#   'scale_pad',
#   'scale_nopad',
#   'narrow_bottle_cap',
#   'wide_bottle_cap',
#   'bottle_nocap',
#   'bottle_cap_on',
#   'bottle_cap_down',
#   'dropper',
#   'red_dropping_bottle',
#   'white_dropping_bottle',
#   'red_line',
#   'blue_line',
#   '90_pipe',
#   'gas_guide_tube',
#   'color_ring',
#   'iron_supporting',
#   'tube_clamp',
#   'tube_hand',
#   '250_beaker',
#   'green_line',
# ]
# basedir = r"/run/media/cv/d/mzt/dataset/tansuanyan/top/simulation"
# voc_ann = os.path.join(basedir, "label")

# 物质的称量和溶液的配置
# top
# classes = [
#     'graduated_cylinder',
#     'scale_nopad',
#     'scale_pad',
#     'spoon',
#     '50_beaker',
#     '250_beaker',
#     'dropper',
#     'glass_rod',
#     'narrow_bottle_cap',
#     'wide_bottle_cap',
#     'bottle_nocap',
#     'green_line',
#     'blue_line',
#     'bottle_cap_on',
#     'bottle_cap_down',
#     'scale_nopad_number'
# ]

# front
# classes = [
#     'graduated_cylinder',
#     'scale_nopad',
#     'scale_pad',
#     'spoon',
#     '50_beaker',
#     '250_beaker',
#     'dropper',
#     'glass_rod',
#     'bottle_cap',
#     'bottle_nocap',
#     'bottle_cap_on',
#     'bottle_cap_down'
# ]
# basedir = r'/run/media/cv/d/mzt/dataset/weight_dissolve/front'
# voc_ann = os.path.join(basedir, "label")


# 密度
# classes = [
#     'dropper',
#     'scale_nopad',
#     # 'scale_nopad_number',
#     'metal_block',
#     'graduated_cylinder',
#     '250_beaker',
#     'face'
# ]
# basedir = r"/run/media/cv/d/mzt/dataset/density/front"
# voc_ann = os.path.join(basedir, "label")

# CO2
# Front
# classes = [ 
#   'conical_flask_off',
#   'conical_flask_on',
#   'gas_jar_cover',
#   'gas_jar_nocover',
#   'bottle_cap',
#   'bottle_nocap',
#   'green_line',
#   'bottle_cap_on',
#   'bottle_cap_off',
#   'long_funnel',
#   '90_pipe',
#   'gas_guide_tube',
#   'wood_strip_off',
#   'wood_strip_on',
#   'tweezer',
#   'alcohol_lamp_off',
#   'alcohol_lamp_on'
#  ]
# base_dir = r'/run/media/cv/d/mzt/dataset/co2/front'

# top
# classes = [ 
#   'conical_flask_off',
#   'conical_flask_on',
#   'matt_gas_jar_cover',
#   'glossy_gas_jar_cover',
#   'gas_jar_nocover',
#   'wide_bottle_cap',
#   'narrow_bottle_cap',
#   'bottle_nocap',
#   'green_line',
#   'blue_line',
#   'bottle_cap_on',
#   'bottle_cap_off',
#   'long_funnel',
#   '90_pipe',
#   'gas_guide_tube',
#   'wood_strip_off',
#   'wood_strip_on',
#   'tweezer',
#   'alcohol_lamp_off',
#   'alcohol_lamp_on'
#  ]

# base_dir = r'/run/media/cv/d/mzt/dataset/co2/top'

# tansuanyan simulation
# front
classes = [
  'tube_plug',
  'tube_noplug',
  'tweezer',
  'scale_pad',
  'scale_nopad',
  'bottle_cap',
  'bottle_nocap',
  'bottle_cap_on',
  'bottle_cap_down',
  'dropper',
  'red_dropping_bottle',
  'white_dropping_bottle',
  '90_pipe',
  'gas_guide_tube',
  'color_ring',
  'iron_supporting',
  'tube_clamp',
  'tube_hand',
  '250_beaker'
]
base_dir = r"/run/media/cv/d/mzt/dataset/tansuanyan/front/simulation"

voc_ann = os.path.join(base_dir, "label")
voc = False
yolo_base = osp.join(base_dir,"yolo")
yolo_label = osp.join(yolo_base, 'labels')
yolo_image = osp.join(yolo_base, 'images')


def convert(size, box):
    dw = 1. / size[0] if size[0] != 0 else 0
    dh = 1. / size[1] if size[1] != 0 else 0
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open(osp.join(voc_ann, image_id + '.xml'))
    out_file = open(osp.join(yolo_label, image_id + '.txt'), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    if not osp.exists(yolo_label):
        os.makedirs(yolo_label)
    
    print(yolo_base)
    if voc:
        for image_set in sets:
            image_ids = open(osp.join(voc_txt, image_set + '.txt')).read().strip().split()
            list_file = open(osp.join(yolo_base, image_set + '.txt'), 'w')
            for image_id in tqdm(image_ids):
                list_file.write(osp.join(yolo_image, image_id + '.jpg') + "\n")
                convert_annotation(image_id)
            list_file.close()
    else:
        for filename in tqdm(os.listdir(voc_ann)):
                image_id = filename[:-4]
                convert_annotation(image_id)