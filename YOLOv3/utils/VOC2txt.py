""""
    利用k-mean对数据集设定适合其anchor-box预测。

    在yolo v3中，有3中尺度预测，每种尺度根据大小赋予其相应大小的anchor-box，即共需要9个anchor；这就决定了k-mean中聚类个数为9类。
"""
import xml.etree.ElementTree as ET
import os
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import matplotlib.patches as patches


# 数据集中类别名称
pacal_voc_classes_Path = open(r'../utils/pascal_voc_classes.json')
classes = list(json.load(pacal_voc_classes_Path))       # voc的类别数
print(classes)

def convert_annotation(image_id):
    in_file = open('E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations\%s.xml'%image_id)        # 打开xml文件

    if not os.path.exists(r'E:\dataset\voc2yolo\img'):
        os.makedirs(r'E:\dataset\voc2yolo\img')

    out_file_img = open(r"E:\dataset\voc2yolo\img\{}.txt".format(image_id),'a')        # 生成图像的txt格式文件
    # out_file_img = open(r"E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\train.txt",'a')
    out_file_label = open(r'E:\dataset\voc2yolo\labels\{}.txt'.format(image_id),'a')         # 生成lable的txt格式文件

    tree = ET.parse(in_file)    # 解析xml文件对象
    root = tree.getroot()       # 获取xml的根节点
    size = root.find('size')    # 在第根节点下寻找size节点

    #   ********** 图片的txt制作 ******************
    voc_img_dir = "E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages\{}.jpg".format(image_id)     # 图片所在的路径
    out_file_img.write(voc_img_dir)     # 将图片的路径写到txt文件
    out_file_img.write("\n")            # 换行

    #   ********* label的txt制作*******************
    img = cv2.imread(voc_img_dir)       # 读取图片
    h,w = img.shape[:2]
    dw = 1. / img.shape[1]              # 图片尺寸的归一化因子
    dh = 1. /img.shape[0]               # 图片尺寸的归一化因子
    # 下面获取相关的box信息
    cnt = len(root.findall('object'))
    if cnt == 0:
        print('nulll null null.....')
        print(image_id)
    cc = 0
    for obj in root.iter('object'):
        cc += 1
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls) +1
        xmlbox = obj.find('bndbox')
        if dw * float(xmlbox.find('xmin').text) < 0. or dw * float(xmlbox.find('xmax').text) < 0. or dh * float(
                xmlbox.find('ymin').text) < 0. or dh * float(xmlbox.find('ymax').text) < 0.:
            print(image_id)

        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))

        out_file_label.write(str(cls_id) + " " + str((b[0] + b[1]) / 2 * dw) + " " + str((b[2] + b[3]) / 2 * dh ) + " " + str(
            (b[1] - b[0]) * dw) + " " + str((b[3] - b[2])*dw))

        # img = plt.imread(voc_img_dir)
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(img)
        # currentAxis = fig.gca()
        # xmin = b[0]* dw * w   # b[0] * w(img.shape(0))
        # ymin = b[2]* dh * h   # b[1] * h
        # width = (b[1]-b[0]) * dw * w
        # hight = (b[3]-b[2])* dh * h
        # rect = patches.Rectangle((xmin, ymin), width,hight, linewidth=1, edgecolor='b',facecolor='none')
        # currentAxis.add_patch(rect)
        # plt.show()

        if cc < cnt:
            out_file_label.write("\n")
    out_file_label.close()
    out_file_img.close()








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_root",type=str,required=False,default=r'E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations')     # xml文件的上一个目录的路径
    parser.add_argument("--img_root",type=str,required=False,default=r'E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages')
    parser.add_argument("--out_ImgRoot",type=str,required=False,default=r'E:\dataset\voc2yolo')    # 生成的图像txt格式文件保存的目录
    parser.add_argument("--out_labelRoot",type=str,required=False,default=r'E:\dataset\voc2yolo')  # 生成的标签txt格式文件保存的目录
    parser.add_argument("--trainTxtPath",type=str,required=False,default=r'E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Main\train.txt')  # 使用train.txt的路径
    arg = parser.parse_args()

    isUseTxt = True         # 是否使用txt中的训练数据作为数据转换，默认是true，用voc的train.txt
    image_ids =[]            # 训练数据的图像名称前缀
    # 如果使用voc的train.txt进行转化，则image_id等于train.txt中的格式
    if isUseTxt:
        # 打开train.txt获取要转换的图像id
        with open(arg.trainTxtPath) as f:
            image_ids.append([line.strip() for line in f.readlines()])

        image_ids = image_ids[0][:]
        # 遍历train.txt中所有的图像id
        for image_id in image_ids:
            convert_annotation(image_id)









