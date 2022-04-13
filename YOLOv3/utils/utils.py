# coding:utf8
from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def to_cpu(tensor):
    return tensor.detach().cpu()

def weights_init_normal(m):
    """
        模型weight初始化
    """
    classname = m.__class__.__name__        # __class__：表示实例对象的类  __name__：名称
    if classname.find("Conv") != -1:        # find()函数是检查一个字符串是否是另一个字符串的字串，并返回字串所在起始位置索引，这里就是最后一个
        torch.nn.init.normal_(m.weight.data,0.0,0.02)       # 初始化
    elif classname.find("BatchNorm2d")  != -1:  # 如果最后一个字符串是BatchNorm2d则先给Conv层初始化，然后再初始化BatchNorm层
        torch.nn.init.normal_(m.weight.data,1.0,0.02)
        torch.nn.init.constant_(m.bias.data,0.0)



def xywh2xyxy(x):
    """
    将坐标为(center_x, center_y, w, h)的格式转换为(x1, y1, x2, y2)的格式
    Args:
        x: 输入的(center_x, center_y , w, h) 格式的坐标
    Returns:
        转换后的(x1, y1, x2, y2)格式的坐标
    """
    y = x.new(x.shape)  # pytorch中new():创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    极大值抑制。移除比conf_thres更低的对象置信度分数的检测，并执行nms进行进一步过滤检测
    Args:
        prediction: 经过model输出后的预测，shape为(batch, num_anchors * grid_size * grid_size *3 , 25)   其中25为(x,y,w,h,conf,cls)
        conf_thres: 置信度阈值
        nms_thres: 非极大值抑制阈值
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # From (center_x,cneter_y,w,h) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])                               # prediction[:, :4]是预测框的xywh。# prediction[..., 4]是pred_conf可信度
    output = [None for _ in range(len(prediction))]                                    # output的列表是有batch个None

    for image_i,image_pred in enumerate(prediction):
        # 得到置信度预测框，过滤掉anchor置信度分数小于阈值的预测框
        # image_pred:是batch中每一个图的预测框的置信度conf。 image_pred.shape(num_anchors * grid_size * grid_size * 3,25)    其中25为（x,y,w,h,conf,cls)
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # 基于anchor的置信度conf过滤完后，看看是否还有保存的预测框，如果都被过滤掉了，则认为没有实体目标被检测到.  则处理下一张图片
        if not image_pred.size(0):
            continue

        # 对象的置信度（conf） * 类别置信度(cls)
        # 计算处理： 先选取每个预测框所代表的最大类别值，再将这个值乘以对应anchor的置信度，这样就将类别预测精准度和置信度都考虑再内。
        # 每个置信预测框都会对应一个score值
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]          # torch.max(x,1)返回两个内容：最大值和最大值索引，所以这里的[0]是指返回最大值
        # 基于score值，将置信预测框从大到小进行排序。 score.shape=(batch * anchor_num * grid * grid *3,1)
        # image_pred = image_pred[[-score).argsort()],troch.sort默认是降序，所以用-score表示升序
        # 置信预测：image_pred ==> (num_anchors * grid_size * grid_size * 3, 25)  25 => (x,y,w,h,conf,cls)。   注意torch.sort返回值有两个：第一个是排序后的数据，第二个是排序后的索引。   这里是排序后的索引
        image_pred = image_pred[torch.sort(-score,dim=0)[1]]
        # image_pred[:, 5:] ==> (num_anchors * grid *grid *3, cls)
        # 该处理是获取每个置信预测框所对应的类别预测分数(class_confs) 和类别索引（class_preds)
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)           # keepdim:主要用于保持矩阵的二位特性
        # 将置信预测框的x, y, w, h, conf,类别预测分值和类别索引关联到一起
        # detections ==> (num_anchor * grid *grid *3, 7)  其中 7 => (x, y, w, h, conf, class_conf, class_pred)
        detections = torch.cat((
                                image_pred[:, :5],          # （x,y,w,h,conf)，并且在上面经过torch.sort进行按大到小排序了
                                class_confs.float(),        # 最大类别分数
                                class_preds.float()),       # 最大类别分数索引
                                1)

        # 执行非极大值抑制，抑制过程是一个迭代-遍历-消除的过程
        # （1）将所有框的得分排序，选中最高分及其对应的框；
        # （2）遍历其余的框，如果和当前最高分框的重叠面积（IOU）大于一定阈值，就将框删除；
        # （3）从未处理的框中继续选一个得分最高的，重复上述过程。
        keep_boxes = []
        while detections.size(0):
            # detections[0, :4]是第一个置信预测框，也是当前序列中分值最大的置信预测框
            # 计算当前序列的第一个（分值最大）置信预测框与整个序列预测框的IOU，并将IOU大于阈值的设置为1，小于的设置为0
            layer_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres         # 因为detections[0,:4]是一个数值，所以会被省略掉一个维度，所以要在第0个维度进行unsqueeze
            # 匹配与当前序列的第一个（分值最大）置信预测框具有相同类别标签的所有预测框（将相同类别标签的预测框标记为1）
            label_match = detections[0,-1] == detections[:, -1]             # detections[0,-1]是一个分类值，例如10。   detections[:,-1] == 10表示最后一个为元素10的就为True，结果label_match是[true,true,false,false,true,,,,]

            # 与当前序列的第一个（分值最大）置信预测框IOU大，说明这些预测框预期相交面积大
            # 如果这些预测框的标签与当前序列的第一个（分值最大）置信预测框的相同，则说明是预测的同一目标
            # 对与当前序列第一个（分值最大）置信预测框预测了同一目标的设置为1（包括当前序列第一个（分值最大）置信预测框本身）
            invalid = layer_overlap & label_match               # label_ouverlap是一个list[True,True,False] 而label_match为[true,true....] 两者相与
            # 取出对应置信预测框的置信度，将置信度作为权重.  当invalid的元素为True就取出权重保存到weight中，当invalid的元素为False不保存
            weights = detections[invalid,4:5]

            # 把预测为同一目标的预测框进行合并，合并后认为是最优的预测框，合并方式如下：
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            # 保存当前序列中最终识别的预测框
            keep_boxes += [detections[0]]
            # ~invalid表示取反，将之前的0变成1，即取剩下的预测框，进行新一轮计算
            detections = detections[~invalid]
        if keep_boxes:
            # 每张图片的最终预测框有pred_boxes_num个，ouput[image_i]的shape为(pred_boxes_num,7)  其中7 => (x,y,w,h,conf,class_conf,class_pred)
            output[image_i] = torch.stack(keep_boxes)

         # (batch_size, pred_boxes_num, 7) 7 =》x,y,w,h,conf,class_conf,class_pred
    return output



def get_batch_statistics(outputs, targets, iou_threshold):
    """
    计算每个样本真阳性、预测分数和预测标签
    Args:
        outputs: 经过模型输出后的预测值。  shape为 (batch_size, pred_boxes_num, 7) 7 =》x,y,w,h,conf,class_conf,class_pred
        targets: 真实标签的值。   shape为(num, 6)  6=>(batch_index, cls, center_x, center_y, widht, height)
        iou_threshold:
    Returns:
        返回的是（[true_positives, pred_score, pred_labels]，[],,,,)

    """
    batch_metrics = []

    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            # 如果第i个output为None，则continue
            continue

        # 注意output和outputs的shape不一样，output少了batch那个维度。output: (pred_boxes_num, 7) 7 =》x,y,w,h,conf,class_conf,class_pred。class_conf:最大类别分数，class_pred:最大类别分数索引（也就是预测的label）
        output = outputs[sample_i]  # 取出outputs的每一个batch

        pred_boxes = output[:, :4]      # 预测框的x,y,w,h
        pred_score = output[:, 4]       # 预测框的置信度
        pred_labels = output[:, -1]     # 预测框的类别label

        # 长度为pred_boxes_num(也就是anchor_num * grid * grid *3)的list，初始化为0，如果预测框和实际框匹配，则设置为1
        true_positives = np.zeros(pred_boxes.shape[0])

        # 获取真实目标的类别label
        annotations = targets[targets[:, 0] == sample_i]            # targets[:, 0]是batch_index。 输出第sample_i个batch的targets
        annotations = annotations[:, 1:] if len(annotations) else []        # annotations.shape=(cls,x,y,w,h)
        targets_label = annotations[:, 0] if len(annotations) else []       # annotations[:, 0]是label

        if len(annotations):    # len(annotations)>0: 表示这张图片有真实的目标框
            detect_boxes = []
            target_boxes = annotations[:, 1:]      # 真实目标框的x,y,w,h

            for pred_i,(pred_box,pred_label) in enumerate(zip(pred_boxes, pred_labels)):    # 遍历所有预测框boxs和labels

                # 如果检测到检测框个数等于真实框个数，就退出
                if len(detect_boxes) == len(annotations):
                    break

                # 如果该预测框的类别标签不存在与目标框的类别标签集合中，则必定是预测错误
                if pred_label not in targets_label:
                    continue

                # 将一个预测框与所有真实目标框做IOU计算，并获取IOU最大的值（iou），和与之对应的真实目标框的索引号（box_index)
                iou, box_index = bbox_iou(pred_box.unsqueeze(0),target_boxes).max(0)        # 预测框和目标框计算的iou中哪个最大，返回最大值和索引
                # 如果最大IOU大于阈值，则认为该真实目标框被发现。注意要防止被重复记录
                if iou >= iou_threshold and box_index not in detect_boxes:
                    true_positives[pred_i] = 1          # 对该预测框设置为1
                    detect_boxes += [box_index]  # 记录被发现的实际框索引号，防止预测框重复标记，即一个实际框只能被一个预测框匹配
        # 保存当前图片被预测的信息
        # true_postives:预测框的正确与否，正确设置为1，错误设置为0；
        # pred_scores: 预测框的置信度;
        # pred_labels： 预测框的类别标签
        batch_metrics.append([true_positives, pred_score, pred_labels])
    return batch_metrics


def ap_per_class(tp,conf,pred_cls,target_cls):
    """
    计算平均精度，给定的recall和precision曲线。
    Args:
        tp（true_postives）:预测框的正确与否，正确设置为1，错误设置为0；（list）
        conf: Objectness value from 0-1.预测框的置信度;(list)      Objectness本质上是物体存在于感兴趣区域内的概率的度量。如果我们Objectness很高，这意味着图像窗口可能包含一个物体
        pred_labels： 预测框的类别标签(list)
        target_cls:True object classes,真正标签的列别(list)
    """

    # 将objectness排序
    i = np.argsort(-conf)  # 升序
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes(去除重复的classes)
    unique_classes = np.unique(target_cls)      # np.unique():该函数是去除数组中的重复数字，并进行排序之后输出。

    # Create Precision-Recall curve and compute AP for each class
    # 创建Precision-Recall 曲线并计算每个类别的AP
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes,desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # 真实目标的数量
        n_p = i.sum()                   # 预测目标的数量

        if n_p == 0 and n_gt == 0 :
            continue
        elif n_p ==0 or n_gt ==0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # 累积FP(false positive,假正）和TP（true positive，真正）
            fpc = (1 - tp[i]).cumsum()          # np.cumsum():函数的功能是返回给定axis上的累计和.axis不给定具体值，就把numpy数组当成一个一维数组。
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)        # recall = tp / 真实目标数量
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)     # Precision = tp / (tp + fp)
            p.append(precision_curve[-1])

            # AP from recall-precison curve
            ap.append(compute_ap(recall_curve,precision_curve))

    # Compoute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2* p * r /(p + r +1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    """
    计算平均精度， 给定的recall和precision曲线
    Args:
        recall: the recall curve(list), recall曲线
        precision: the precision curve(list), precision曲线
    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation(计算正确的AP)
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))       # numpy.concatenate((a1,a2,…), axis=0)函数，能够一次完成多个数组的拼接
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size -1, 0, -1):
        mpre[i -1] = np.maximum(mpre[i -1], mpre[i])        # np.maximum是用来求最大值的，这个最大值的shape依据最大的array的shape来定。比如比较（2，），（3,2）的数组的最大值。那么maximum返回的array的shape就是（3,2）

    # to calculate area under RP curve,look for points
    # where X axis(recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum(\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap




























def bbox_wh_iou(wh1,wh2):
    """
    计算在feature map的真实框的wh和预设锚框在feature map的wh之间的iou
    Args:
        wh1:预设锚框在feature map的wh
        wh2:在feature map的真实框的wh

    return:
        两个框的wh之间的iou，和boxes的iou不同，这个是求长宽之间的iou
    """
    wh2 = wh2.t()   # 转置，将n行2列的变成2行n列，这样返回的iou就是一个计算1个预设框和多个真实框的iou列表
    w1,h1 = wh1[0],wh1[1]   # 分别获取预设锚框的w和h
    w2,h2 = wh2[0],wh2[1]   # 分别后去真实框的w和h

    inter_area = torch.min(w1,w2) * torch.min(h1,h2)    # 内部的面积
    union_area = (w1 * h1 +1e-16) + w2 * h2 - inter_area    # 两个框加起来的总面积
    return inter_area / union_area


def bbox_iou(box1,box2,x1y1x2y2=True):
    """
        返回两个bounding box的IOU
    """
    if not x1y1x2y2:
        # 如果不是x1y1x2y2这种格式，就进行转换。从center_x,center_y,w,h  ==>  x1y1x2y2
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    else:
        # 边界框的坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]


    # 获取相交矩形的坐标
    inter_rect_x1 = torch.max(b1_x1,b2_x1)
    inter_rect_y1 = torch.max(b1_y1,b2_y1)
    inter_rect_x2 = torch.min(b1_x2,b2_x2)
    inter_rect_y2 = torch.min(b1_y2,b2_y2)
    # 两个box的交集的面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1,min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 +1 ,min=0)
    # 两个box的总面积
    b1_area = (b1_x2 - b1_x1 +1) * (b1_y2 - b1_y1 +1)
    b2_area = (b2_x2 - b2_x1 +1) * (b2_y2 - b2_y1 +1)

    # 求iou， iou=(box1 ∩ box2) / (box1 + box2 -box1 ∩ boxes2)
    iou = inter_area / (b1_area + b2_area -inter_area + 1e-16)
    return iou









def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thresh):
    # pred_boxes => (batch,anchor_num,gride,gride,4)
    # pred_cls => (batch,anchor_num,gride,gride,80)
    # targets => (num,6)     其中6 => (batch_index,cls,center_x,center_y,width,height)
    # anchors  => (3,2)

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)     # batch的数量，表示batch中的第b张图片
    nA = pred_boxes.size(1)     # anchor的数量
    nC = pred_cls.size(-1)    # class的数量 => 80
    nG = pred_boxes.size(2)     # gride，也就是特征图的大小

    # Output tensor,建立输出的tensor并初始化
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)     # (batch,anchor_num,gride,gride)    tensor.fill_(value):将tensor中的所有值都填充为指定的value,这里是全部都指定为0
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)   # (batch,anchor_num,gride,gride)，并初始化全部为1
    class_mask = FloatTensor(nB, nA,nG,nG).fill_(0)  # (batch,anchor_num,gride,gride)，并初始化全部为0
    iou_scores = FloatTensor(nB,nA,nG,nG).fill_(0)  # (batch,anchor_num,gride,gride)
    tx = FloatTensor(nB,nA,nG,nG).fill_(0)          # # (batch,anchor_num,gride,gride)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)     # (batch,anchor_num,gride,gride,class_num)


    # 转换box的相对位置
    # 这一步是将x,y,w,h这四个归一化的变量变成真正的尺寸，因为当前图像的尺寸是nG，所以要乘以nG，变成feature map的尺寸
    target_boxes = target[:, 2:6] * nG       # (num,4)     4 => (cneter_x,center_y,width,height)
    gxy = target_boxes[:, :2]         # (num,2)  获取center_x,center_y
    gwh = target_boxes[:, 2:]         # (num,2)  获取h,w

    # 获取最佳的anchors
    # 这一步是为每一个目标从三种anchors框中分配一个最后的框
    # anchor是设置的锚框，gwh是真实框在feature map上的映射，这里比较两者的交集，选出最佳的锚框，因为只是选择那种锚框，不用考虑中心坐标
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])    # 遍历所有的anchors（3个），分别计算和真实框的iou,ious.shape=(3,num) num表示有n个实际的框，
    # ious(3,num),该处理是为每一个目标框选取一个IOU最大的anchors框，best_ious表示最大的iou的值，best_n表示最大IOU对应anchor的index
    best_ious,best_n = ious.max(0)  # best_ious 和 best_n 的长度均为num，best_n是num个目标框对应的anchor索引， 例如best_n=[2,1,0,1,,,,,n] 每个目标框都有和anchor的iou最大,返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）

    # 分离target的值
    # .t()表示转置，（num，2） => (2,num)
    # (2,num)    2 =>(batch_index,cls) =>b(num)表示对应num个index，target_labels(num)表示对应num个labels
    b, target_labels = target[:, :2].long().t()  # b代表一个batch的图片索引，target_labels代表target中的标签，例如：[0,0,0,0,0,1,1,1]， ？？？.long()是否已经把tensor类型变成int类型了
    gx, gy = gxy.t()    # gx代表num个x，gy代表num个y； 也就是把x和y都统计为一行；这是一个batch中所有图像的x,y
    gw, gh = gwh.t()    # 把一个batch中所有图像的w，h都统计为一行
    gi, gj = gxy.long().t()     # .long()是把浮点型转化为整型（去尾），这样就可以得到目标框中心点所在的网格坐标，gi,gj的shape=（num,1)
    print("gi",gi)


    # ---------------得到target实体框obj_mask和target非实体框noobj_mask   start-----------------------
    # 设置mask
    # 表示batch中的第b张图片，其网格坐标为（gj，gi）的单元网格存在目标框的中心点，该目标框所匹配的最优anchor索引为best_n
    obj_mask[b,best_n,gj,gi] = 1    # obj_mask.shape=(batch,anchor_num,grid,grid)  对目标实体框中心点所在的单元网格(best_n)的索引，其最优anchor设置为1,在含有best_n的锚点的feature map相应的gi,gj位置置1，其余为0
    noobj_mask[b,best_n,gj,gi] = 0  # 对target实体框中心点所在的单元网格，其最优的anchor设置为0； obj_mask[batch,[1,1,0,0,2],[3,2,13,15,5],12] = 1就相当于[batch,1,3,12],[batch,1,2,12],[batch,0,13,12]...

    # 当iou超过ignore阈值时，将noobj_mask设置为零                                                     iou=( anchor1+目标框1的iou      anchor1+目标框2的iou    anchor1+目标3的iou  。。。
    # ious.t():(3,num)  => (num,3)  就是将三个锚点中的每个anchor与真实框的iou统计一起                          anchor2+目标框1的iou       anchor2+目标框2的iou    anchor2+目标3的iou 。。。
    # 这里不同于上一个策略，上个策略是找到与目标框最优的anchors框，每个目标框对应一个anchor框                         anchor3+目标1的iou         anchor3+目标2的iou      anchor3+目标3的iou 。。。）
    # 这里不考虑最优问题，只要目标框与anchor的iou大于阈值，就认为是有效的anchor框，即noobj_mask对应位置设置为0
    for i,anchor_ious in enumerate(ious.t()):       # i就是目标框的个数
        noobj_mask[b[i],anchor_ious > ignore_thresh,gj[i],gi[i]] = 0

    # 以上操作得到了target实体框obj_mask和target非实体框noobj_mask，target实体框是与实体一一对应的，一个实体有一个最匹配的目标框；
    # 目标实体框noobj_mask，该框既不是实体最匹配的，而且还要该框与实体IOU小于阈值，这也是为了让正负样本更加明显。
    # --------------------------得到target实体框obj_mask和目标非实体框noobj_mask    end-----------------------


    # --------------------------得到target实体框的归一化坐标(tx,ty,tw,th)   start----------------------------
    # 将x,y,w,h重新归一化
    # 注意：要知道为什么这样子，此处的归一化和传入target的归一化方式不一样，
    # 传入的target的归一化是实际x,y,w,h / img_size,即实际x,y,w,h在img_size中的比例
    # 此处的归一化中，中心坐标x,y是基于单元网络的，w,h是基于anchor框，此处归一化的x,y,w,h，也是模型要拟合的值
    #a = gx-gx.floor()
    tx[b,best_n,gj,gi] = gx - gx.floor()           # 得到相对于feature map左上角坐标的偏移量
    ty[b,best_n,gj,gi] = gy - gy.floor()            # floor():向下取整
    # width and height                                                          # anchors.shape=(3,2)
    tw[b,best_n,gj,gi] = torch.log(gw / anchors[best_n][:,0] + 1e-16)           # anchors[best_n][:,0]首先看后面[:,0]先，所有行第0列，然后挑选[best_n]的.如：a=[[1,2],[3,4],[5,6]]   则 a[0,1,0,1,1][:,0]=[1,3,1,3,3]
    th[b,best_n,gj,gi] = torch.log(gh / anchors[best_n][:,1] + 1e-16)
    # ---------------------------得到target实体框的归一化坐标（tx, ty, tw, th）  end---------------------------------

    # One-hot encoding of label
    # 表示batch中的第b张图片，其网格坐标为(gj,gi)的单元网格存在目标框的中心点，该目标框所匹配的最优anchor索引为best_n,其类别为target_labels
    tcls[b,best_n,gj,gi,target_labels] = 1          # tcls.shape=(batch,anchor_num,gride,gride,class_num)

    # 计算标签正确性和最好的anchor
    # class_mask:将预测正确的标记为1（正确的预测了实体中心点所在网格坐标，哪个anchor框可以最匹配实体，以及实体的类别）
    #print("判断最大索引和target_label是否相等:",pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels)
    try:
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()      # 预测的最大值的索引 == 标签的label,pred_cls.shape=(b,anchor_num,grid,grid,class),而pred_cls[b, best_n, gj, gi] =[0,1,2,3.....class]
    except:
        print("预测的argmax(-1)的返回值和target_label的值不同，或者需要将target_label转化为int")
    # class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == int(target_labels)).float()    这里需要把label转化为int类型吗??因为在dataset中label好像不是int类型？？？？？？？？？？？

    # iou_scores:预测框pred_boxes中的正确框与target实体框target_boxes的交集IOU，以IOU作为分数，IOU越大，分值越高
    iou_scores[b,best_n,gj,gi] = bbox_iou(pred_boxes[b,best_n,gj,gi],target_boxes,x1y1x2y2=False)
    # tconf: 正确的目标实体框，其对应anchor框的置信度为1，即置信度的标签，这里转为float，是为了后面和预测的置信度值做loss计算
    tconf = obj_mask.float()

    # iou_scores：预测框pred_boxes中的正确框与目标实体框target_boxes的交集IOU，以IOU作为分数，IOU越大，分值越高。
    # class_mask：将预测正确的标记为1（正确的预测了实体中心点所在的网格坐标，哪个anchor框可以最匹配实体，以及实体的类别）
    # obj_mask：将目标实体框所对应的anchor标记为1，目标实体框所对应的anchor与实体一一对应的
    # noobj_mask：将所有与目标实体框IOU小于某一阈值的anchor标记为1
    # tx, ty, tw, th： 需要拟合目标实体框的坐标和尺寸
    # tcls：目标实体框的所属类别
    # tconf：所有anchor的目标置信度
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf



