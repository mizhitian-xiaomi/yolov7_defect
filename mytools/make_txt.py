import os
from os import listdir
import random


classes = [ 'rolled-in_scale', 'patches', 'crazing', 'pitted_surface', 'inclusion', 'scratches' ]
base_dir = r"/home/mzt/dataset/NEU-DET/yolo/"

label_dir = os.path.join(base_dir, "labels")
image_dir = os.path.join(base_dir, "images")

train_text = os.path.join(base_dir, "train.txt")
test_text = os.path.join(base_dir, "test.txt")
val_text = os.path.join(base_dir, "val.txt")

def count_label(file_path):
    print(f"------ Count From: {file_path}")
    label_cnt = {t:0 for t in classes}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        label_path = line.strip().replace('jpg', 'txt').replace('images', 'labels')
        with open(label_path, 'r') as txt:
            txt_lines = txt.readlines()
        for txt_line in txt_lines:
            label_idx = int(txt_line.split(" ")[0])
            try:
                label_cnt[classes[label_idx]] += 1
            except Exception as e:
                print(f"Error: {e} --- {line.strip()} | {label_idx}")
    print(f"------ sum = {len(lines)}")
    for k, v in label_cnt.items():
        print(f"{k:20}\t{v:5}")
    return label_cnt
        

def make_txt_by_video(train_video, val_video, sign_idx):
    ftest = open(test_text, 'w')
    ftrain = open(train_text, 'w')
    fval = open(val_text, 'w')

    total_xml = os.listdir(label_dir)
    train, val, test = 0, 0, 0
    for filename in total_xml:
        name = os.path.join(image_dir, filename.replace("txt","jpg")) + '\n'
        video_name = filename.split('_')[sign_idx]
        if video_name in train_video:
            ftrain.write(name)
            train += 1
        elif video_name in val_video:
            fval.write(name)
            val += 1
        else:
            ftest.write(name)
            test += 1
    ftrain.close()
    fval.close()
    ftest.close()

    print(f"train = {train}")
    print(f"test = {test}")
    print(f"val = {val}")


def for_yolo_by_video(keep=True, sign_idx=0):
    train_list = []
    val_list = []

    file_list = listdir(label_dir)
    video_set = list(set([name.split('_')[sign_idx] for name in file_list]))
    if not keep or not os.path.exists(test_text):
        train_num = int(0.7 * len(video_set))  # 训练集数量
        val_num = int(0.5 * (len(video_set) - train_num))  # 验证集数量
        # val_num = int(len(video_set) - train_num)  # 验证集数量
        random.shuffle(video_set)
        train_list = video_set[:train_num]  # 训练集视频数量
        val_list = video_set[train_num:train_num + val_num]  # 验证集视频数量
        print("------ new test ------")
        make_txt_by_video(train_video=train_list, val_video=val_list, sign_idx=sign_idx)
    else:
        with open(test_text, 'r') as f:
            test_list = list(set(os.path.basename(line).strip().split("_")[sign_idx] for line in f.readlines()))
            test_num = len(test_list)
        train_num = int(0.8 * (len(video_set) - test_num))  # 训练集视频数量
        val_num = len(video_set) - train_num - test_num  # 验证集视频数量
        random.shuffle(video_set)
        for video_id in video_set:
            if video_id in test_list:
                continue
            if train_num > 0:
                train_num -= 1
                train_list.append(video_id)
            elif val_num > 0:
                val_num -= 1
                val_list.append(video_id)
        print("------ keep test ------")
        make_txt_by_video(train_video=train_list, val_video=val_list, sign_idx=sign_idx)


def for_yolo_by_random(keep):

    train_list = []
    val_list = []
    xml_num = len(os.listdir(label_dir))
    file_list = [ os.path.join(image_dir,filename.replace("txt","jpg"))+"\n" for filename in os.listdir(label_dir)]
    random.shuffle(file_list)
    
    if not keep:
        train_num = int(0.8 * xml_num)  # 训练集数量
        test_num = int((xml_num - train_num) * 0.6)  # 测试集数量
        # test_num = 0  # 测试集数量
        val_num = xml_num-train_num-test_num  # 验证集数量
        
        train_list = file_list[:train_num]
        test_list = file_list[train_num:train_num+test_num]
        val_list = file_list[-val_num:]
        
        with open(test_text, 'w+') as f:
            f.writelines(test_list)

    else:
        with open(test_text, 'r') as f:
            test_list = f.readlines()
        test_num = len(test_list) # 测试集数量
        train_num = int(0.95 * (xml_num-test_num))  # 训练集数量
        val_num = xml_num-train_num-test_num  # 验证集数量
        
        for filename in file_list:
            image_file = os.path.join(image_dir,filename.replace("txt","jpg"))
            if image_file in test_list:
                continue
            if len(train_list) <= train_num:
                train_list.append(image_file)
            else:
                val_list.append(image_file)

        
    print("train = ", len(train_list))
    print("test = ", len(test_list))
    print("val = ", len(val_list))

    with open(train_text, 'w+') as f:
        f.writelines(train_list)

    with open(val_text, 'w+') as f:
        f.writelines(val_list)


if __name__ == '__main__':
    print(f"------ Make From: {base_dir}")
    # for_yolo_by_video(True, sign_idx=0)
    # for_yolo_by_random(True)
    
    label_cnt = count_label(train_text)
    label_cnt = count_label(val_text)
    label_cnt = count_label(test_text)
    