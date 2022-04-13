# -*- coding: utf-8 -*-

from __future__ import division

from utils.models import Darknet
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def evaluate(model, root, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    """
    在验证时评价指标
    Args:
        model: 模型
        path: 验证集数据root路径
        iou_thres: 真实框和预测框iou阈值
        conf_thres: 置信度阈值（置信度等于包含物体的概率 * 物体分类的概率）
        nms_thres: 非极大值抑制的阈值
        img_size: 图像的尺寸
        batch_size: batch尺寸
    Returns:
    """
    model.eval()        # 评价使用

    # Get dataLoader
    dataset = VOCData(root,img_size=img_size, trainFileName='val', augment=False, multiscale=True, normalized_labels=True)
    dataLoder = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []     # List of tuples(TP, confs, pred), 验证样本的评价指标
    for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataLoder, desc="Detecting objects")):
        # img:(batch,c,h,w)
        # target:(batch_index,class, center_x, center_y, w, h)

        # 提取标签
        labels += targets[:,1].tolist()         # 将一个batch中的所有target中的class取出来，并用一个列表保存。 num个cls
        # 重新缩放target
        #将一个batch中所有的target的xywh即（num,center_x,center_y,w,h)，把x，y，w，h格式转换为x1,y1,x2,y2的格斯，注意，转换后该格式依旧为归一化的值
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        # 将归一化的x1, y1, x2, y2 转为真实坐标,之前归一化是除以原图的h和w，所以要乘回来
        targets[:, 2:] *= img_size

        imgs = Variable(imgs,requires_grad=False)       # 验证时requires_grad为False

        with torch.no_grad():
            # 上下文管理器，被该语句warp(包裹)起来的部分不会track(跟踪)梯度
            # outputs的shape为(batch, num_anchors * grid_size * grid_size *3 , classes_num),num_anchor代表有多少个锚框（3），因为有3个尺度的yolo层，所以最后输出的ouputs是num_anchors * grid * grid * 3
            outputs = model(imgs)
            # 对输出output进行非极大值抑制得到新的output，它的shape为（batch, pred_boxes_num, 7) 其中的7为（x,y,w,h,conf,class_conf,class_pred）   class_conf:最大类别分数，class_pred:最大类别分数索引（也就是预测的label）
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # outputs:(batch,pred_boxes_num,7)   7 => (x,y,w,h,conf,class_conf,class_pred)
        # target:(num,6)   6 => (batch_index, class, center_x, center_y, w, h)
        # 计算每个样本真阳性、预测分数和预测标签。返回：[（true_postive,pred_scores,pred_labels)...] 其中true_postives:预测框的正确与否，正确设置为1，错误设置为0；pred_scores: 预测框的置信度;pred_labels： 预测框的类别标签
        print("targets",targets)
        sample_metrics += get_batch_statistics(outputs, targets,iou_thres)
    print("sample_metrics",sample_metrics)

    # 将所有图片的预测信息进行concatenate,每张图片包含了true_positives, pred_scores, pred_labels 这三个信息
    #true_positives, pred_scores, pred_labels = [np.concatenate(x,0) for x in list(zip(*sample_metrics))]    # zip(*）解包，得到（true_postive1,true_positive2,....)  (pred_scores1,pred_scores2....)   (pred_labels1,pred_labels2....)
    #precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision,recall,AP,f1,ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3_voc.cfg", help="path to model definition file")
    parser.add_argument("--root", type=str, default="E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Main", help="验证集root路径")
    parser.add_argument("--weights_path", type=str, default="weights/darknet53.conv.74", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/voc.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 验证集root路径
    root = opt.root
    json_file = open(r'utils/pascal_voc_classes.json')
    class_name = json.load(json_file)


    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # 如果weight的名称是以weights结尾的,就使用darknet中的方法load_darknet_weights，否则使用load_state_dict方法加载weight
        model.load_darknet_weight(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weight_path))

    print("Compute AP")

    precison, recall, AP, f1, ap_class = evaluate(
        model,
        root=root,
        iou_thres=opt.iou_thres,
        conf_thres=opt.ocnf.thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )
    print("Average Precisions(平均精确度）:")
    for i, c in enumerate(ap_class):
        print("+ Class '{}'  ({}) - AP: {}".format(c,class_name[c],AP[i]))

    print("mAP:{}".format(AP.mean()))