import os
import shutil
import xml.etree.ElementTree as ET

def count_voc_label(xml_file, targets):
    if isinstance(targets, str):
        targets = [targets]
    targets_dict = dict()
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall("object"):
        if member[0].text in targets:
            if targets_dict.get(member[0].text) is None:
                targets_dict[member[0].text] = 1
            else:
                targets_dict[member[0].text] += 1
    return targets_dict

def count_yolo_label(txt_file, targets, classes):
    if isinstance(targets, str):
        targets = [targets]
    targets_dict = dict()
    with open(txt_file, mode="r", encoding="utf-8")as f:
        for line in f.readlines():
            name = classes[int(line.split(" ")[0])]
            if name in targets:
                if targets_dict.get(name) is None:
                    targets_dict[name] = 1
                else:
                    targets_dict[name] += 1
    return targets_dict

def get_dstfile(file_path, insert=-3):
    temp = file_path.split("/")
    temp.insert(insert, "detect_vs_gt")
    dst_file_path = "/".join(temp)
    if not os.path.exists(os.path.split(dst_file_path)[0]):
        os.makedirs(os.path.split(dst_file_path)[0])
    return dst_file_path

voc_label_dir = r"/home/mzt/dataset/co2/20211013-补充小木条/front/label"
yolo_label_dir = r"/home/mzt/work_dirs/co2/detect/front-train/labels"
classes = [ 
  'conical_flask_off',
  'conical_flask_on',
  'gas_jar_cover',
  'gas_jar_nocover',
  'bottle_cap',
  'bottle_nocap',
  'green_line',
  'bottle_cap_on',
  'bottle_cap_off',
  'long_funnel',
  '90_pipe',
  'gas_guide_tube',
  'wood_strip_off',
  'wood_strip_on',
  'tweezer',
  'alcohol_lamp_off',
  'alcohol_lamp_on'
 ]
for filename in os.listdir(voc_label_dir):
    xml_file = os.path.join(voc_label_dir, filename)
    txt_file = os.path.join(yolo_label_dir, filename.replace("xml","txt"))
    if not os.path.exists(txt_file):
        continue
    detect_image_file = os.path.join(os.path.split(yolo_label_dir)[0],filename.replace("xml","jpg"))
    voc_label_cnt = count_voc_label(xml_file, ["wood_strip_on", "wood_strip_off"])
    yolo_label_cnt = count_yolo_label(txt_file, ["wood_strip_on", "wood_strip_off"], classes)
    
    flag = True
    for k in voc_label_cnt.keys():
        nvoc = voc_label_cnt[k]
        nyolo = yolo_label_cnt[k] if yolo_label_cnt.get(k) else 0
        
        if nvoc != nyolo:
            flag = False
    if not flag:
        dst_label_path = get_dstfile(xml_file)
        dst_image_path = dst_label_path.replace("label","image").replace("xml","jpg")
        if not os.path.exists(os.path.split(dst_image_path)[0]):
            os.makedirs(os.path.split(dst_image_path)[0])
        shutil.copy(xml_file, dst_label_path)
        shutil.copy(detect_image_file, dst_image_path)
        print(dst_image_path)
