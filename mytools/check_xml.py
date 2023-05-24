import os
import xml.etree.ElementTree as ET

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

label_dir = os.path.join(base_dir, "label")
image_dir = os.path.join(base_dir, "images")

# images_dir = os.path.join('/run/media/cv/d/mzt/dataset/dianxue/top/yolov5', 'images')
# image_dir = images_dir


def check_by_listdir():
    print("check_by_listdir: ", label_dir, " is testing")
    calculate = dict.fromkeys(classes, 0)
    # cnt = 0
    for file in os.listdir(label_dir):
        if (file[len(file) - 4:] == '.xml'):
            annotate_path = os.path.join(label_dir, file)
            try:
                tree = ET.parse(annotate_path)
                root = tree.getroot()
            except Exception as e:
                print(f"Error: {e} --- {file}")
            for member in root.findall("object"):
                value = member[0].text
                if value in classes:
                    calculate[value] += 1
                if value not in classes:
                    print("#" + member[0].text + "#")
                    print(file)
    for k, v in calculate.items():
        print(f"{k:20}\t{v:5}")
    # print(cnt)


def delete():
    print(f"delete: {label_dir} is deleting...")

    if not (os.path.exists(label_dir) and os.path.exists(image_dir)):
        return

    for label in os.listdir(label_dir):
        (filepath, tempfilename) = os.path.split(label)
        (filename, extension) = os.path.splitext(tempfilename)
        labelname = label_dir + '/' + label
        imagename = image_dir + '/' + filename + '.jpg'
        if not os.path.exists(imagename):
            print('image ---- ', imagename)
            os.remove(labelname)
    print("label finished")
    for image in os.listdir(image_dir):
        (filepath, tempfilename) = os.path.split(image)
        (filename, extension) = os.path.splitext(tempfilename)
        imagename = image_dir + '/' + image
        labelname = label_dir + '/' + filename + '.xml'
        if not os.path.exists(labelname):
            print("label ---- ", labelname)
            os.remove(imagename)
    print("image finished")


def clean():
    print(f"clean: {label_dir} is cleaning...")
    for file in os.listdir(label_dir):
        filename = label_dir + '/' + file
        ext = filename[len(filename) - 4:]
        flag = False
        if (ext == '.xml'):
            try:
                tree = ET.parse(filename)
                root = tree.getroot()
            except Exception as e:
                print(f"Error: {e} --- {file}")
            for member in root.findall("object"):
                value = member[0].text
                member[0].text = value.strip()
                if not value in classes:
                    root.remove(member)
                # if value == 'ed_dropping_bottle':
                #     root.remove(member)
                    # member[0].text = 'red_dropping_bottle'
                    # print(file)
                # elif value == "250_beake":
                #     member[0].text = '250_beaker'
                # elif value == "dropperw":
                #     member[0].text = 'dropper'
                # elif value=='bottle_cap_dowm':
                #     member[0].text = 'bottle_cap_down'
            filepath = os.path.join(label_dir, file)
            tree.write(filepath)
    # print(wrong_label)
    print("finished")


if __name__ == '__main__':
    # clean()
    # print("\n"+"="*50)
    # delete()
    print("\n"+"="*50)
    check_by_listdir()
