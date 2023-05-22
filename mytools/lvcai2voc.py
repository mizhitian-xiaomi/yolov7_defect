import json
import os


def write_xml(xml_file, image_path, bbox_list):
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
            xml.write(f"\t<object>\n")
            label = bbox[0]
            xmin = bbox[1][0]
            ymin = bbox[1][1]
            xmax = bbox[1][2]
            ymax = bbox[1][3]
            xml.write(f"\t\t<name>{label}</name>\n")
            xml.write(f"\t\t<pose>Unspecified</pose>\n")
            xml.writelines([f"\t\t<truncated>0</truncated>\n",
                           f"\t\t<difficult>0</difficult>\n"])
            xml.write(f"\t\t<bndbox>\n")
            xml.write(f"\t\t\t<xmin>{xmin}</xmin>\n")
            xml.write(f"\t\t\t<ymin>{ymin}</ymin>\n")
            xml.write(f"\t\t\t<xmax>{xmax}</xmax>\n")
            xml.write(f"\t\t\t<ymax>{ymax}</ymax>\n")
            xml.write(f"\t\t</bndbox>\n")
            xml.write(f"\t</object>\n")
        xml.write(f"</annotation>")


IMAGE_WIDTH = 2560
IMAGE_HEIGHT = 1920

# base_dir = r"/home/mzt/dataset/lvcai/guangdong_round2_train_201810111/guangdong_round2_train_20181011/单瑕疵图片"
# for subdir in os.listdir(base_dir):
#     data_dir = os.path.join(base_dir, subdir)
#     if not os.path.isdir(data_dir):
#         continue
data_dir = r"/home/mzt/dataset/lvcai/guangdong_round2_train_20181011/多瑕疵图片"
for filename in os.listdir(data_dir):
    if not filename.endswith("json"):
        continue
    json_file = os.path.join(data_dir, filename)
    with open(json_file, mode="r", encoding='utf-8')as f:
        json_data = json.load(f)
    bbox_list = []
    for item in json_data['shapes']:
        label = item['label']
        points = item['points']
        (xmax, ymax) = points[:][-2]
        (xmin, ymin) = points[:][0]
        bbox = [label, (xmin, ymin, xmax, ymax)]
        bbox_list.append(bbox)

    xml_file = os.path.join(data_dir, filename.replace("json", "xml"))
    image_path = os.path.join(data_dir, filename.replace("json", "jpg"))
    write_xml(xml_file, image_path, bbox_list)
