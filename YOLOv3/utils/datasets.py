import os
import sys
import glob

import cv2
import numpy as np
import random
from PIL import Image
from PIL import ImageFile                       # ImageFile模块为图像的打开和保存提供了一些函数。
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import xml.etree.ElementTree as ET                          # 解析xml文件

import torch
import torch.nn.functional as F                 #
# from augmentations import horisontal_flip       # 用于数据增强，图像水平翻转
from torch.utils.data import Dataset
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True          # 当用Image.open()打开jpeg图片遇到截断时，程序会跳过去读取另一张图片


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    print(h,w)
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)      # (左边填充数， 右边填充数， 上边填充数， 下边填充数)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)      # 用value的值填充,pad:扩充维度，用于预先定义出某维度上的扩充参数； mode:constant、reflect、replicate三种模式，分别表示常量、反射、复制

    #print(pad)
    return img, pad



class VOCData(Dataset):
    def __init__(self,root,img_size=416,trainFileName = 'train',augment=False,multiscale=True,normalized_labels=True):
        '''
            读取VOC格式的数据，从root路径加载图片和标签
        Args:
            root:voc主路径
            img_size:将图片统一缩放到的大小，注意要是32的倍数
            trainFileName:判断是用train还是val数据集,默认是’train'
            augment:增强，默认true
            multiscale：多尺度检测，默认true
            nomalized_labels:标签进行归一化，默认true

        Return:
            img(tensor格式的，[B,C,H,W]）
        '''
        self.root = root            # voc数据集的主路径
        self.img_size = img_size    # 图片缩放的大小，32的倍数
        self.max_objects = 20       # 检测的最大目标数
        self.augment = augment      # 数据增强
        self.normalized_labels = normalized_labels  # 标签进行归一化
        self.multiscale = multiscale                # 是否进行多尺度
        self.batch_count = 0                        # 计算到第几个batch
        self.txtPath = os.path.join(self.root, "ImageSets", "Main", trainFileName + '.txt')     # train.txt文件所在路径

        # print(self.txtPath)

        with open(self.txtPath) as f:
            self.imgID = [x.strip() for x in f.readlines()]      # self.imgID:保存的是图片的名称前缀
        # print(len(self.imgID))
        assert len(self.imgID) != 0,'txt文件打开出错！请检查路径或txt文件为空'


        # 打开voc_classe的json文件
        try:
            json_file = open('utils/pascal_voc_classes.json','r')        # 以只读方式打开
            self.class_dict = json.load(json_file)                          # class_dict:保存类别的dict，{类别名:编号}
            json_file.close()                                               # 关闭json文件
        except Exception as E:  # 当出现异常
            print("json文件加载出错")
            exit(-1)

    def __getitem__(self,idex):
        # ---------
        #  image
        # ---------
        # 读取图片
        self.imgPath = os.path.join(self.root,"JPEGImages",self.imgID[idex]+'.jpg')      # 图片的路径
        print(self.imgPath)
        img = transforms.ToTensor()(Image.open(self.imgPath).convert('RGB'))    # 读取图片，并转化成Tensor格式,(c,h,w)
        # print(img.shape)

        if len(img.shape) != 3:        # 如果图片不是3维的，就在第0个维度增加一个维度
            img = img.unsqueeze(0)
            img = img.expand((3,img.shape[1:]))

        _,h,w = img.shape               # 没有将h和w填充之前的图片原尺寸
        h_factor,w_factor = (h,w) if self.normalized_labels else (1,1)      # 如果使用label归一化，则h_factor=h,后面将框的边长/h_factor得到归一化的高；如果不使用归一化，则边长/1

        # 下面先将图片填充到h和w相等，在进行resize，这样做可以尽量保持数据完整
        img,pad = pad_to_square(img,0)        # pad_to_square:将图片填充到h和w相等
        _,padded_h,padded_w = img.shape       # 这里是填充后的图片的大小，h，w

        img = F.interpolate(img.unsqueeze(0), size=self.img_size, mode="nearest").squeeze(0)    # 利用插值的方法，对输入的张量数组进行上下采样操作，合理的改变数组尺寸大小,变成416*416
        _,resize_h,resize_w = img.shape

        # resize后的缩放比例
        scale_h = resize_h / padded_h
        scale_w = resize_w / padded_w

        # ---------
        #   label
        # ---------
        # 从xml读取标签
        test = []
        boxes = []  # 保存bbox坐标，{x,y,w,h}
        labels = []  # 保存类别名，并将json文件的字典的值代替
        labelPath = os.path.join(self.root,"Annotations",self.imgID[idex]+'.xml')       # xml文件路径

        tree = ET.parse(labelPath)              # 创建一个解析xml的对象，输入标签的路径,返回根节点所在位置
        for obj in tree.findall('object'):      # 找到二级节点，并遍历所有的object节点
            obj_struct = {}                     # 建立一个字典，保存name和坐标信息
            obj_struct['name'] = self.class_dict[obj.find('name').text]        # 获取所有的‘name’属性,并且用json文件里面字典中对应的值替代
            bbox = obj.find('bndbox')           # 找到三级节点bndbox
            # obj_struct['bbox'] = [int(bbox.find('xmin').text),      # 找到‘xmin’,键值是'bbox'
            #                       int(bbox.find('ymin').text),      # 找到‘ymin’,键值是'bbox'
            #                       int(bbox.find('xmax').text),      # 找到‘xmax',键值是'bbox'
            #                       int(bbox.find('ymax').text)]      # 找到’ymax‘,键值是'bbox'

            # labels.append(obj_struct)   # labels是以字典的形式保存标签名称+坐标

            xmin,ymin,xmax,ymax= int(bbox.find('xmin').text),int(bbox.find('ymin').text), int(bbox.find('xmax').text),int(bbox.find('ymax').text)
            # 之前填充是将（最长边 - 最短边） / 2 ,相当于把短边移动到 + 填充值位置的一半，所以将xmin等加上填充的位置就等于填充后目标框的位置
            xmin = (pad[0] + xmin) * scale_w        # xmin是在416*416这个尺度
            ymin = (pad[2] + ymin) * scale_h
            xmax = (pad[1] + xmax) * scale_w
            ymax = (pad[3] + ymax) * scale_h

            # 再将加上填充后的坐标归一化成yolo格式的坐标  center_x_norm,center_y_norm,w,h,
            center_x = (xmin + xmax) /2
            center_y = (ymin + ymax) /2
            center_x_norm = center_x / resize_w
            center_y_norm = center_y / resize_h
            w_norm = (xmax - xmin) / resize_w
            h_norm = (ymax - ymin) / resize_h
            #center_x_norm = ((xmin + xmax) /2) / padded_w
            #center_y_norm = ((ymin + ymax) /2) / padded_h
            #w_norm = (xmax - xmin) / padded_w
            #h_norm = (ymax - ymin) / padded_h

            labels.append(obj_struct['name'])
            boxes.append([center_x_norm, center_y_norm, w_norm, h_norm])

            test.append([obj_struct['name'],center_x_norm,center_y_norm,w_norm,h_norm])     # 这个只是再测试的时候用



        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # FloatTensor = torch.FloatTensor
        target = torch.zeros((len(boxes),6))
        target[:,2:] = (torch.from_numpy(np.array(boxes)))
        target[:,1] = (torch.from_numpy(np.array(labels)))
        #target[:,2:] = FloatTensor(boxes)
        #target[:,1] = FloatTensor(labels)




        # 如果图片要进行增强
        # if self.augment:
        #     if np.random.random()<0.5:
        #         img,target = horisontal_flip(img,target)

        print(target)
        return img,target


        # img = img.permute([1,2,0])
        # plt.imshow(img)
        # ax = plt.gca()
        # ax.add_patch(plt.Rectangle((target1[1,1], target1[1,2]), (target1[1,3]-target1[1,1]), (target1[1,4]-target1[1,2]), color="blue", fill=False, linewidth=1))
        # ax.add_patch(plt.Rectangle((target1[2, 1], target1[2, 2]), (target1[2, 3] - target1[2, 1]), (target1[2, 4] - target1[2, 2]),color="blue", fill=False, linewidth=1))
        # ax.add_patch(plt.Rectangle((target1[0, 1], target1[0, 2]), (target1[0, 3] - target1[0, 1]), (target1[0, 4] - target1[0, 2]),color="blue", fill=False, linewidth=1))
        # ax.add_patch(plt.Rectangle((target1[3, 1], target1[3, 2]), (target1[3, 3] - target1[3, 1]),(target1[3, 4] - target1[3, 2]), color="blue", fill=False, linewidth=1))
        # ax.add_patch(plt.Rectangle((target1[4, 1], target1[4, 2]), (target1[4, 3] - target1[4, 1]),(target1[4, 4] - target1[4, 2]), color="blue", fill=False, linewidth=1))
        # plt.show()


    def __len__(self):
        return len(self.imgID)  # 返回训练集中共有多少个数据


    def collate_fn(self,batch):
        # 在我们的图像大小不同时，需要自定义函数callate_fn来将batch个图像整理成统一大小的，若读取的数据有(img, box, label)
        # 这种你也需要自定义，因为默认只能处理(img, label)

        img,targets = list(zip(*batch))      # 将__getitem__返回的数据组成的batch解压，组成一个batch为：(img1,target1),(img2,target2)........
                                            # zip(*) 解压后为：,(img1,img2,img3.....),(targets1,targets2,target3......)]

        # 有可能__getitem__返回的图像返回的图像是None，所以要过滤掉
        targets = [boxes for boxes in targets if boxes is not None]
        # print("first-----",target)

        # boxes是每张图像上的目标，但是每个图片上目标框数量不一样，所以需要给这些框添加索引，对应到是哪个图像上的框
        for i,boxes in enumerate(targets):           # enumerate()可以修改（target）中的值
            boxes[:,0] = i                          # 给一个批次中每个图片中的目标框标上索引，说明这个bbox是属于这个batch中哪张图片的
        targets = torch.cat(targets,0)                # 把target拼接起来
        #target = torch.Tensor(target)
        #target = torch.stack(targets,0)
        #print("secen",targets)

        # 每十个批次选择一个新的图像大小
        # if self.multiscale and self.batch_count % 10 == 0：
        #     self.img_size = random.choice(range(self.min_size,self.max_size+1,32))
        # # 每个图像大小都不同，调整图像大小
        # imgs = torch.stack([resize(img,self.img_size) for img in imgs])

        imgs = torch.stack(img,0)       # 将图像进行拼接
        # print(imgs.shape)
        return imgs,targets




    # 根据坐标作图
    def plot_pic(self):
        img_path = self.imgPath
        labels = self.box

        img = plt.imread(img_path)
        h,w,c = img.shape
        # print(h,w,c)

        coords = []
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img)
        currentAxis = fig.gca()
        for label in labels:
            xmin = float(label[1])
            ymin = float(label[2])
            xmax = float(label[3])
            ymax = float(label[4])

            width = ymax - ymin
            height = xmax - xmin

            rect = patches.Rectangle((xmin,ymin),width,height,linewidth=1,edgecolor='r',facecolor=None)
            currentAxis.add_patch(rect)
            plt.show()














#class yoloData(Dataset):
    # """
    #     读取yolo格式的数据
    # """




#data = VOCData(r'E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012')
#a = data.__getitem__(3)
#b= data.__getitem__(2)
# data.plot_pic()


#data.collate_fn([a,b])



