# -*- coding: utf-8 -*-
from __future__ import division

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets
from utils.utils import to_cpu



def create_modules(module_defs):
    """
        函数功能：根据传入的网络模型部分创建网络，根据module_defs中的module配置构造层块的module list
        Args:
            module_defs:从yolo.cfg文件读取到的网络模型，形式为：[{'type':'convolution','batch_norm':1。。。。。。}，{。。。。。}]等由字典组成的列表
        return:返回一个超参和网络列表
    """
    hyperparams = module_defs.pop(0)    # 从yolo.cfg文件获取超参，在第一个字典{‘type'：net，........}中,list.pop(index):删除index项的数据，并返回index项数据，这里就module_defs删除了超参那项。
    # print(hyperparams)
    # print(module_defs)
    output_filters = [int(hyperparams['channels'])]     # 输出的filters = 超参中key为’channel'这一项的值,这里是3
    # print("output_filter",output_filters)
    module_list = nn.ModuleList()       # 和nn.sequential()不同的是nn.ModeuleList():可以把任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，方法和 Python 自带的 list 一样
    for module_i,module_def in enumerate(module_defs):       # 遍历'convolution'的所有层
        modules = nn.Sequential()       # nn.Sequential():不同于 nn.ModuleList，它已经实现的 forward 函数，而且里面的模块是按照顺序进行排列的，所以我们必须确保前一个模块的输出大小和下一个模块的输入大小是一致的

        if module_def['type'] == 'convolutional':     # 提取出'convolution'的信息
            bn = int(module_def["batch_normalize"])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2
            modules.add_module(         # nn.Sequential()有add_module方法添加模块
                "conv_{}".format(module_i),      # name
                nn.Conv2d(
                    in_channels = output_filters[-1],     # 输入的维度
                    out_channels = filters,            # 输出的维度
                    kernel_size = kernel_size,       # 卷积核的size
                    stride = int(module_def['stride']),   # 步长
                    padding = pad,
                    bias = not bn,
                ),
            )
            if bn:      # 如果是batch_normalize，就添加一个bn层
                modules.add_module("batch_norm_{}".format(module_i),nn.BatchNorm2d(filters,momentum=0.9,eps=1e-5))      # eps:epsillon
            if module_def['activation'] == 'leaky':      # 在convolution中如果是用leaky做激活函数，就使用leaky激活层
                modules.add_module("leaky_{}".format(module_i),nn.LeakyReLU(0.1))



        elif module_def['type'] == 'maxpool':       # 如果当key为'type'的值为'maxpool'的时候，创建maxpool层(这里不需要）
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module("_debug_padding_{}".format(module_i), nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module("maxpool_{}".format(module_i), maxpool)


        elif module_def['type'] == 'upsample':      # 如果当key为”type“的值为‘upsample’的时候，将该字典保存的模块用来创建upsamle块
            upsample = Upsample(scale_factor=int(module_def['stride']),mode='nearest')
            modules.add_module("upsamle_{}".format(module_i),upsample)


        # 路由层（Route）：参数有1或2个值。当只有一个值，输出这一层通过该值索引的特征图。当有两个值时，它将返回由这两个值索引的拼接特征图。如-1和61，因此该层级将输出从前一层级(-1)到第61层的特征图，并将它们按深度拼接
        # 该层先由一个空的模块进行占位，最后通过forward进行对特征图拼接(route或shortcut残差连接）
        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]      # module_def可能有两个值,所以要通过split(',')拆分开
            filters = sum([output_filters[1:][i] for i in layers])          # 因为是会将两个特征图进行拼接，所以filter就等于两个的维度相加
            #print("filters",filters)
            modules.add_module("route_{}".format(module_i),EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[1:][int(module_def['from'])]
            modules.add_module("shortcut_{}".format(module_i),EmptyLayer())

        elif module_def['type'] == 'yolo':
            # yolo层级对应检测层级。参数anchor定义9组锚点，但是它们只是由mask标签使用的属性所索引的锚点。mask=0，1，2表示使用第（1，2，3）锚点。而掩膜表示检测层的每一个单元预测3个框，检测共有3层，分别对应不同尺度
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # 摘取anchors，即选择那几个anchor的w和h，anchors里面有2*9个数，表示共有9个不同尺寸的(w,h)anchor
            anchors = [int(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]      # anchor[i],anchor[i+1]表示(w,h)
            anchors = [anchors[i] for i in anchor_idxs]     # 选出3个anchor，每个anchor都有(w,h)
            num_class = int(module_def['classes'])          # 记录类别数Num_class
            img_size = int(hyperparams['height'])           # 输入的图像大小，这里是416

            # 因为yolo有3个层，所以有3个定义detection层，每一层都有3个anchor送到YOLOLayer中
            yolo_layer = YOLOLayer(anchors,num_class ,img_size)      # anchor：输入anchor的高宽；img_size:图片的尺寸,
            modules.add_module("yolo_{}".format(module_i), yolo_layer)
        module_list.append(modules)
        output_filters.append(filters)  # 将新创建的块的输出的channel添加到output_filters列表中


   #  print(output_filters)
    #print(module_list)
    # 返回一个超参和网络列表
    return hyperparams,module_list




class Upsample(nn.Module):
    def __init__(self,scale_factor,mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self,x):
        x = F.interpolate(x,scale_factor=self.scale_factor,mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """ route 和  shorcut 图层的占位符 """
    def __init__(self):
        super().__init__()


class YOLOLayer(nn.Module):
    """
        这是定义yolo的检测层
    """
    def __init__(self,anchors,num_classes,img_dim=416):
        """
        YOLO层是在Darknet层之后的，所以它的forward输入的是feature map而不是images
        Args:
            anchors: 1个尺度的3个锚点,每个锚点还包括(w,h)
            num_classes: 分类数量
            img_dim: 图像的输入大小
        """
        super().__init__()
        self.anchors = anchors              # 3个锚点
        self.num_classes = num_classes      # 分类数量
        self.num_anchors = len(anchors)     # 锚点的数量（这里是3个）
        self.ignore_thres = 0.5             # 忽视的阈值，就是说小于0.5的框忽略
        self.mse_loss = nn.MSELoss()        # 使用MSELoss（均方差）作为损失函数，用在边界框回归损失
        self.bce_loss = nn.BCELoss()        # 使用BCELoss（二值交叉熵）作为损失函数，用在分类损失
        self.obj_scale = 1                  # 当含有目标的时候，Iobj的值=1，就是包含目标的损失函数前面的系数
        self.noobj_scale = 100              # 当不包含有目标的时候，Inoobj的值=100，就是不包含目标的损失函数前面的系数
        self.metrics = {}                   # 评价指标，这里是一个字典
        self.img_dim = img_dim              # 输入图像的大小，这里是416
        self.grid_size = 0                  # 将图像划分成几个grid_cell的size



    def compute_grid_offsets(self,grid_size,cuda=True):
        """
        Args:
            grid_size:特征图的大小
            cuda: 是否使用cuda
        return:
        """
        self.grid_size = grid_size  # 将当前的grid_size添加到self中的grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size      # 从原图变成特征图缩小多少倍，也就是步幅，416 / grid_size

        # 计算每个grid_cell的偏移
        # 对于grid_size=13,下面是在x，y维度上创建13 x 13 的网格。grid_x,grid_y (1,1,gride,gride)
        self.grid_x = torch.arange(g).repeat(g,1).view([1,1,g,g]).type(FloatTensor)   # torch.repeat()沿着指定的维度重复tensor。不同与expand()，本函数复制的是tensor中的数据,第一个参数表示的是复制后的列数，第二个参数表示复制后的行数。
        self.grid_y = torch.arange(g).repeat(g,1).t().view([1,1,g,g]).type(FloatTensor)

        # 图片缩小多少倍，对应的anchor也要缩小相应的倍数
        self.scaled_anchors = FloatTensor([(a_w / self.stride,a_h / self.stride) for a_w,a_h in self.anchors])      # 将3个anchors根据从原图缩放到特征图的倍数缩放锚点，anchors是3个锚点，(w,h)

        # scaled_anchors.shape(3,2),3个anchors，每个anchor都有(w,h)两个量，下面步骤是把这两个量划分开
        self.anchors_w = self.scaled_anchors[:,:1].view((1,self.num_anchors,1,1))       # 获取w的参数，并reshape成(1,3,1,1)
        self.anchors_h = self.scaled_anchors[:,1:2].view((1,self.num_anchors,1,1))      # 获取h的参数，并reshape成(1,3,1,1)




    def forward(self,x,target=None,img_dim=None):
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        self.img_dim = img_dim      # img_dim=416,特征图像的大小，因为有3个yolo层，所以不指定feature map的大小，输入在darknet的forward中输入img_dim
        num_sample = x.size(0)      # num_sample保存的是输入训练数据的batch数据；   输入的数据的shape = (batch,c,w,h)
        grid_size = x.size(2)       # 正向传播特征图的大小，等于w，也就是把原图分成w*h的grid cell
        print("feature map size(grid_Size):",grid_size)
        prediction = (
            x.view(num_sample,self.num_anchors,5 + self.num_classes,grid_size,grid_size)          # 将经过darknet获取的shape(batch,c,w,h)的feature map形状重新定义成(batch,anchors,num_class+5,grid_size,grid_size)
            .permute(0, 1, 3, 4, 2)         # 调换维度，最后输出的shape为（batch,num_anchors,grid_size,grid_size,classes+5=85）  (batch,3,13,13,85)
            .contiguous()       # torch.view等方法操作需要连续的Tensor。transpose、permute 操作虽然没有修改底层一维数组，但是新建了一份Tensor元信息，并在新的元信息中的 重新指定 stride。torch.view 方法约定了不修改数组本身，只是使用新的形状查看数据。如果我们在 transpose、permute 操作后执行 view，Pytorch 会抛出以下错误
        )

        # 获得输出，这里的prediction是初步的所有预测，在grid_size*grid_size个网格中，它表示每个网络都会有num_anchors(3)个anchors框
        # x,y,w,h,pred_conf的shanpe都是一样的，(batch,num_anchor,grid_size,grid_size)
        x = torch.sigmoid(prediction[...,0])    # x是经过prediction输出后经sigmoid的center x的偏移，也就是tx.      切片操作中，...用于替代多维度。
        y = torch.sigmoid(prediction[...,1])    # y是经过prediction输出后经sigmoid的center y 的偏移，也就是ty.   (batch,anchor_num,grid,grid)
        w = prediction[...,2]       # w是经过prediction输出后的width，这里不用sigmoid函数控制在0-1之间            tw
        h = prediction[...,3]       # h是经过prediction输出后的height。                                       th
        pred_conf = torch.sigmoid(prediction[..., 4])    # pred_conf是包含有物体的概率
        pred_cls = torch.sigmoid(prediction[..., 5:])     # pred_cls是物体类别的分类概率

        # 如果网格大小与当前不匹配，我们将计算新的偏移量,判断正向传播时特征图的大小是否等于self.grid_size。需要注意的是这里的输入的size是32的倍数
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size,cuda=x.is_cuda)

        # 添加偏移并且缩放的anchors
        pred_boxes = FloatTensor(prediction[...,:4].shape)      # prediction前4个数，也就是boxes的(x,y,w,h)
        # 针对每个网格(feature map)的偏移量，每个网格的单位长度为1，而预测的中心点(x,y)是归一化的(0,1之间），所以可以直接相加
        pred_boxes[...,0] = x.data + self.grid_x       # 这里是公式求预测的偏移量 bx = sigmoid(tx) + cx;       tx = x.data,  cx=grid_x,这里的cx是feature map的所有grid cell
        pred_boxes[...,1] = y.data + self.grid_y       # 预测xy的shape=(1,1,grid,grid)
        pred_boxes[...,2] = torch.exp(w.data) * self.anchors_w      # bw = pw * exp(tw)， pw是预设锚框的宽， 因为是在feature map求的，所以pw是缩放到feature map图的宽self.anchor_w
        pred_boxes[...,3] = torch.exp(h.data) * self.anchors_h      # 预测hw的shape=（1，3，1，1）

        # 将预测的输出pred_boxes、pred_conf重新reshape成(batch,num_anchors * grid_size,85)
        output = torch.cat(
            (# 将pred_boxes重新resahpe成(batch,num_anchors * grid_size * grid_size,4）  -- 边框
            pred_boxes.view(num_sample,-1,4) * self.stride,      # 将在feature map上预测到的锚框 放大到原图最初输入的尺寸
            # 将pred_conf重新reshape成（batch,num_anchors * grid_size * grid_size,1)   -- 可信度
            pred_conf.view(num_sample,-1,1),
            # 将pred_cls重新reshape成(batch,num_anchor * grid_size * grid_size,80)   -- 类别
            pred_cls.view(num_sample,-1,self.num_classes),
            ),
            -1,   # 在最后一个维度上进行拼接，4+1+80
        )

        # 如果输入标签为None，直接返回
        if target is None:
            return output,0
        else:
            # pred_boxes => (batch,anchor_num,grid,grid,4)
            # pred_cls => (batch,anchor,grid,grid,80)
            # targets => (num,6)  其中6 => (batch_index,cls,center_x,center_y,width,height)   batch_index:一个batch中的其中一幅图像的名称索引
            # scale_anchor => (3,2)   一个尺度有3个锚框，每个锚框都有(w,h)
            iou_score,class_mask,obj_mask,noobj_mask,tx,ty,tw,th,tcls,tconf = build_targets(
                pred_boxes = pred_boxes,
                pred_cls = pred_cls,
                target = target,
                anchors = self.scaled_anchors,      # 在feature map中的3个锚点，
                ignore_thresh=self.ignore_thres,
            )
            # iou_score:预测框pred_boxes中的正确框与目标实体框target_box的交集IOU，以IOU作为分数，IOU越大，分值越高；
            # class_mask:将预测正确的标记为1（正确的预测了实体中心点所在的网格坐标，哪个anchor框可以最匹配实体，以及实体的类别）
            # obj_mask:将目标实体框所对应的anchor标记为1，目标实体框所对应的anchor与实体一一对应；
            # noobj_mask:将所有与目标实体框IOU小于某一阈值的anchor标记为1
            # tx,ty,tw,th: 需要拟合目标实体框的坐标和尺寸
            # tcls： 目标实体框的所属类别
            # tconf: 所有anchor的目标置信度   = class * obj
            # ------------------- 上面都是将target进行编码 ---------------------------------end

            # 上面计算到的iou_scores,class_mask,obj_mask,noobj_mask,tx,ty,tw,th和tconf都是(batch,anchor_num,gride,gride)
            # 预测的x,y,w,h,pred_conf也都是(bachch,anchor_num,gride,gride)
            # tcls 和 pred_cls都是(batch,anchor_num,gride,gride,num_class)


            # ------------------- 计算Loss ---------------------------------------------
            # Loss:忽略输出不存在的对象,只计算含有目标的损失（除了conf.loss)
            # 坐标和尺寸的loss计算
            obj_mask = obj_mask.bool()          # convert int8 to bool
            noobj_mask = noobj_mask.bool()      # convert int8 to bool
            loss_x = self.mse_loss(x[obj_mask],tx[obj_mask])        # 这里的损失都是在feature map的尺度下求偏移量的损失
            loss_y = self.mse_loss(y[obj_mask],ty[obj_mask])        # 都是用均方差
            loss_w = self.mse_loss(w[obj_mask],tw[obj_mask])        # w是预测的高，即pred的tx，tw是真实框归一化后的tw
            loss_h = self.mse_loss(h[obj_mask],th[obj_mask])

            # anchor的置信度的loss计算
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask],tconf[obj_mask])  # 含有物体的置信度，只由负责（IOU较大）预测的哪个bounding box的置信度才会计入误差。这里用交叉熵损失
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask],tconf[noobj_mask])    # 不含有物体的置信度（应该输出尽可能低的置信度）
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj  # 总的置信度，包含物体的置信度 * 系数1 + 不包含物体的置信度 * 系数2

            # 类别loss计算
            loss_cls = self.bce_loss(pred_cls[obj_mask],tcls[obj_mask])     # 类别，用交叉熵损失。当第i个grid cell的第i个anchor box负责某一个真实目标时，那么这个anchor box所产生的bounding box才会去计算分类损失函数

            # loss汇总
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics(评价指标）
            cls_acc = 100 * class_mask[obj_mask].mean()     # 分类精确度
            conf_obj = pred_conf[obj_mask].mean()           # 包含物体的平均置信度
            conf_noobj = pred_conf[noobj_mask].mean()       # 不包含物体的平均置信度
            conf50 = (pred_conf > 0.5).float()              # 置信度大于0.5
            iou50 = (iou_score > 0.5).float()               # iou分数大于0.5
            iou75 = (iou_score > 0.75).float()              # iou分数大于0.75
            detected_mask = conf50 * class_mask * tconf     # class_mask:将预测正确的标记为1（正确预测了实体中心点所在网格坐标，哪个anchor框最匹配，以及实体的类别）.shape=(batch,anchor_num,grid,grid)

            obj_mask = obj_mask.float()                     # obj_mask是对目标实体框中心点所在的单元网格(best_n)的索引，其最优anchor设置为1。(batch,anchor_num,grid,grid)

            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)   # 精确度
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)   # 召回率
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)   # 召回率


            # 评价指标
            self.metrics = {
                "loss": to_cpu(total_loss).item(),      # loss:总的loss损失
                "x": to_cpu(loss_x).item(),             # x的坐标损失，item是得到一个元素张量里面的元素值
                "y": to_cpu(loss_y).item(),             # y的坐标损失
                "w": to_cpu(loss_w).item(),             # w的坐标损失
                "h": to_cpu(loss_h).item(),             # h的坐标损失
                "conf": to_cpu(loss_conf).item(),           # 置信度损失
                "cls": to_cpu(loss_cls).item(),             # 类别损失
                "cls_acc": to_cpu(cls_acc).item(),          #
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }


            # 返回经过yolo层输出的数据，shape=(batch,anchor_num,grid,grid,class),和总的loss
            return output,total_loss



class Darknet(nn.Module):
    """
        定义Darknek，并从../config/yolov3.cfg文件读取网络
    """
    def __init__(self,config_path,img_size=416):
        """
        Args:
            config_path:yolo 中darknet的网络结构文件yolov3.cfg的路径
            img_size: 图像的尺寸
        """
        super().__init__()
        self.module_defs = parse_model_config(config_path)      # 从yolov3.cfg文件读取网络结构,此时只有各个模块，还未建立成网络模型
        #self.hyperparams,self.module_list = create_modules(self.module_defs)   # 将上面获取到的网络模块部分用来创建网络模型，返回超参和module_list
        self.hyperparams,self.module_list = create_modules(self.module_defs)    # hyperarams:超参   module_list:从cfg读取的网络模型
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0],"metrics")]      # hasattr():判断对象是否包含对应的属性。layer[0]是Sequential的第一个网络，并判断是否有'metrics"属性
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0,self.seen, 0],dtype=np.int32)

    def forward(self, x, target=None):
        img_dim = x.shape[2]            # 图像的宽高
        loss =0
        layer_outputs, yolo_outputs = [], []        # layer_outputs:卷积层输出的特征        yolo_ouputs:输出的是预测信息shape=(batch,anchor_num,grid,grid,class),和总的loss

        # 一层层遍历网络
        for i,(module_def,module) in enumerate(zip(self.module_defs,self.module_list)):         # module_defs:从cfg读取并解析出来的字典；module_list:创建的module网络
            if module_def['type'] in ["convolutional", "upsample", "maxpool"]:      # 如果'type'是convolutional、upsamle、maxpool的一种，就求特征图
                x = module(x)       # 正向传播
            elif module_def['type'] == "route":                                     # 如果‘type“是’route'（路由层），即把两个特征图进行深度拼接
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")],1)   # 从cfg文件中的route读取layer，如layer=-1，61，并把两个特征图进行拼接
            elif module_def['type'] == 'shortcut':          # Res-net中的残差连接
                layer_i = int(module_def['from'])           # module_def['from'] = -3
                x = layer_outputs[-1] + layer_outputs[layer_i]  # 残差相连
            elif module_def['type'] == 'yolo':                  # yolo层
                x,layer_loss = module[0](x,target,img_dim)      # x输出的是预测结果，yolo层输出的[bx，by，bw，bh，conf,class]和loss
                loss += layer_loss                              # 3个yolo层的总的loss
                yolo_outputs.append(x)                          # 将3个yolo层输出的预测结果保存到yolo_outputs中

            # 保存每一块的特征图
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs,1))        # 将3个yolo层的预测结果在channel上进行拼接
        return yolo_outputs if target is None else (loss,yolo_outputs)      # 如果target不为None则返回3个层的预测结果，否则返回(loss,3个ylo层的预测结果）


    # 加载darknet的权重,输入：weight_path:darknet权重文件路径
    def load_darknet_weight(self,weight_path):
        # 打开权重文件
        with open(weight_path,'rb') as f:
            header = np.fromfile(f,dtype=np.int32,count=5)      # 前5个值是header值
            self.header_info = header                           # 保存权重时需要写header
            self.seen = header[3]                               # number of images seen during training
            weights = np.fromfile(f,dtype=np.float32)           # 其余的都是权重

        # 建立加载backbone权重
        cutoff = None
        if "darknet53.conv.74" in weight_path:                  # 如果文件的名称是darknet53.conv.74，就让cutoff=75
            cutoff = 75

        ptr = 0
        for i,(module_def,module) in enumerate(zip(self.module_defs,self.module_list)):         # self.module_defs是网络配置文件   self.module_list是网络模型
            if i == cutoff:
                break           # 在到达75层的时候退出

            if module_def['type'] == "convolutional":
                conv_layer = module[0]          # 这里的module[0]代表convolutional这一模块里面的conv层
                if module_def["batch_normalize"]:
                    # 加载BN bias,weights,均值和方差
                    # BN层的输出Y与输入X之间的关系是：Y = (X - running_mean) / sqrt(running_var + eps) * weight + bias
                    bn_layer = module[1]        # module[1]代表convolutional这一模块里面的bn层
                    num_b = bn_layer.bias.numel()       # Number of biases,统计一个bn层有多少个bias；   numel() 计算矩阵中元素的个数,获取tensor中一共包含多少个元素
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)          # view_as(tensor):返回被视作与给定的tensor相同大小的原tensor，等效于view(tensor.size())
                    bn_layer.bias.data.copy_(bn_b)              # 将从weights加载到的bias数据赋给bn层的bias
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr+num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                else:
                    # 加载 conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr+num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b


    def save_darknet_weights(self,path,cutoff=-1):
        """
            保存darknet的权重
        Args:
            path:保存权重的路径
            cutoff:保存在0到cutoff之间的层（cutoff=-1代表保存所有层）
        return:
        """
        fp = open(path,"wb")
        self.header_info[3] = self.seen
        self.heager_info.tofile(fp)

        # 遍历所有的层
        for i,(module_def,module) in enumerate(zip(self.module_defs[:cutoff],self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]          # module[0]这是convolutional模块下的conv层
                # 如果有batch norm,则首先保存bn层
                if module_def["batch_normalize"]:
                    bn_layer = module[0]        # module[1]是convolutional模块下的bn层
                    bn_layer.bias.data.cpu().numpy().tofile(fp)         # 将bn层的bias保存到文件夹中   np.tofile(frame):将numpy数据保存到文件中
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)

                # 保存conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # 保存conv层的权重(weights)
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()


# x = cv2.imread(r"E:\data\1.png")
# x = cv2.resize(x,(416,416))
# tran = torchvision.transforms.ToTensor()
# a = tran(x).unsqueeze(0)
# print(a.shape)
# b=np.array([[0,14,0.524,0.573529411764706,0.836,0.666]])
# target = torch.from_numpy(b)
# #print(target)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# a = Variable(a.to(device))
# target = Variable(target.to(device))
# D = Darknet('../config/yolov3_voc.cfg').to(device)
# out = D(a,target)

#D.load_darknet_weight(r'E:\github代码\PyTorch-YOLOv3-master\weights\yolov3_weights_pytorch.pth')

