import os
import os.path as osp
from tqdm import *

        
def txt_convert_xml(txt_file, xml_dir, classes):
    with open(txt_file,"r") as f:
        txt_msg = f.readlines()
    bbox_list = []
    for msg in txt_msg:
        # box = [ float(t) for t in msg.strip().split(" ") ]
        box = [ float(t) for t in msg.strip().split(" ") ]
        label = classes[int(box[0])]
        score = box[-1]
        (x1,y1,x2,y2) = box[1:5]
        # (x,y,w,h) = box[1:5]
        # x1 = int((x-w/2)*IMAGE_WIDTH)
        # y1 = int((y-h/2)*IMAGE_HEIGHT)
        # x2 = int((x+w/2)*IMAGE_WIDTH)
        # y2 = int((y+h/2)*IMAGE_HEIGHT)
        bbox_list.append([label, score, (int(x1), int(y1), int(x2), int(y2))])

    file_name = os.path.basename(txt_file)
    path = osp.join(xml_dir.replace("label","image"), f'{file_name[:-4]}.jpg')
    xml_file = osp.join(xml_dir, file_name.replace("txt", "xml"))
    # 写入文件
    write_xml(xml_file, path,bbox_list) 
    
    
def write_xml(xml_file,image_path,bbox_list):
    with open(xml_file, "w", encoding="utf-8") as xml:
        xml.write(f"<annotation>\n")
        xml.write(f"\t<folder></folder>\n")
        xml.write(f"\t<filename>{os.path.basename(image_path)}</filename>\n")
        xml.write(f"\t<path>{image_path}</path>\n")
        xml.write(f"\t<source>\n")
        xml.write(f"\t\t<database>Unknown</database>\n")
        xml.write(f"\t</source>\n")
        xml.write(f"\t<size>\n")
        xml.write(f"\t\t<width>{IMAGE_WIDTH}</width>\n")
        xml.write(f"\t\t<height>{IMAGE_HEIGHT}</height>\n")
        xml.write(f"\t\t<depth>3</depth>\n")
        xml.write(f"\t</size>\n")
        xml.write(f"\t<segmented>0</segmented>\n")
        for bbox in bbox_list:
            score = bbox[1]
            if score < 0.8:
                continue
            label = bbox[0]
            xmin = bbox[2][0]
            ymin = bbox[2][1]
            xmax = bbox[2][2]
            ymax = bbox[2][3]
            xml.write(f"\t<object>\n")
            xml.write(f"\t\t<name>{label}</name>\n")
            xml.write(f"\t\t<pose>Unspecified</pose>\n")
            xml.writelines([f"\t\t<truncated>0</truncated>\n", f"\t\t<difficult>0</difficult>\n"])
            xml.write(f"\t\t<bndbox>\n")
            xml.write(f"\t\t\t<xmin>{xmin}</xmin>\n")
            xml.write(f"\t\t\t<ymin>{ymin}</ymin>\n")
            xml.write(f"\t\t\t<xmax>{xmax}</xmax>\n")
            xml.write(f"\t\t\t<ymax>{ymax}</ymax>\n")
            xml.write(f"\t\t</bndbox>\n")
            xml.write(f"\t</object>\n")
        xml.write(f"</annotation>")


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

if __name__=="__main__":
    
    txt_dir = r"/run/media/cv/d/mzt/projects/yolov7/density-label/top-bottle_cap_down-xyxy/labels"
    xml_dir = r"/run/media/cv/d/mzt/dataset/weight_dissolve/video/top/bottle_cap_down/label"
    # 称量溶解 top
    classes = [
    'graduated_cylinder',
    'scale_nopad',
    'scale_pad',
    'spoon',
    '50_beaker',
    '250_beaker',
    'dropper',
    'glass_rod',
    'narrow_bottle_cap',
    'wide_bottle_cap',
    'bottle_nocap',
    'green_line',
    'blue_line',
    'bottle_cap_on',
    'bottle_cap_down',
    'scale_nopad_number'
]
    print(txt_dir)
    for file_name in tqdm(os.listdir(txt_dir)):
        txt_file = os.path.join(txt_dir, file_name)
        txt_convert_xml(txt_file,xml_dir, classes)