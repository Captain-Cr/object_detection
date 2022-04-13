from __future__ import division

from utils.models import *
# from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

# from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="训练周期")
    parser.add_argument("--batch_size", type=int, default=8, help="图像batch大小")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3_voc.cfg", help="yolov3.cfg文件的路径")
    #parser.add_argument("--data_config", type=str, default=r"E:\github代码\PyTorch-YOLOv3-master\config\yolov3.cfg", help="yolov3.cfg文件的路径")
    parser.add_argument("--train_path",type=str,default=r'E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012',help="训练数据的路径")
    #parser.add_argument("--pretrained_weights", type=str,default="weights/yolov3_weights_pytorch.pth", help="预训练模型")
    parser.add_argument("--pretrained_weights",type=str,default="weights/darknet53.conv.74",help="预训练模型权重")
    parser.add_argument("--n_cpu", type=int, default=8, help="批处理训练期间要使用的cpu数量")
    parser.add_argument("--img_size", type=int, default=416, help="图像的尺寸")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--root", type=str, default="E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Main", help="验证集root路径")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # 获取data配置,针对coco数据集
    # data_config = parse_data_config(opt.data_config)
    # train_path = data_config["train"]             # 训练数据的路径
    # valid_path = data_config["valid"]             # valid数据的路径
    # class_names = load_classes(data_config["name"])   # 类别名称
    # 针对自己的数据集，在dataset中已经进行加载路径了，所以只需在自己的数据集VOCData(r'E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012')

    # 初始化模型
    model = Darknet(opt.model_def).to(device)       # 将yolov3.cfg里面的层的参数加载到Darknet中
    model.apply(weights_init_normal)                # model.apply(fn)会递归将函数fn应用到父模块的每个子模块submodule,也包括model这个父模块自身

    # If specified we start from checkpoint
    if opt.pretrained_weights:          # 如果输入的预训练权重是以pth结尾的就用load_state_dict()加载预训练权重，否则使用model文件中的load_darknet_weights()方法
        if opt.pretrained_weights.endswith('pth'):
            model.load_state_dict(torch.load(opt.pretrained_weights))       # 加载预训练模型
        else:
            model.load_darknet_weight(opt.pretrained_weights)


    # Get dataloader
    # 通过voc_json文件获取class_name
    json_file = open(r'utils/pascal_voc_classes.json')
    class_name = json.load(json_file)

    dataset = VOCData(opt.train_path,trainFileName = 'train',augment=False,multiscale=True,normalized_labels=True)
    dataLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        )
    optimizer = torch.optim.Adam(model.parameters())      # 使用Adam优化器优化模型里面的参数
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]


    # 开始训练
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()        # 开始时间
        for batch_i,(imgs,targets) in enumerate(dataLoader):
            batches_done = len(dataLoader) * epoch + batch_i            # len(dataLoader)等于dataset/batch
            # img -> (batch,channel,h,w)
            # target -> (num,6)    6->(batch_index,class,center_x_norm,center_y_norm,w_norm,h_norm)
            imgs = Variable(imgs.to(device))

            targets = Variable(targets.to(device), requires_grad=False)

            loss,ouput = model(imgs,targets)        # 经过Darknet输出yolo层的分数和loss
            loss.backward()                         # 反向传播

            if batches_done % opt.gradient_accumulations:
                # 在每步之前累积梯度
                optimizer.step()            # 开始反向传播
                optimizer.zero_grad()       # 清零

            # ---------------------
            #     Log progress(日志进度）
            # ---------------------
            log_str = "\n-------[Epoch %d / %d,Batch %d / %d] ----\n"%(epoch,opt.epochs,batch_i,len(dataLoader))

            # Yolo各层metrics日志
            for i,metric in enumerate(metrics):
                formats = {m:"%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric,0) for yolo in model.yolo_layers]

                # tensorboard的日志
                tensorboard_log = []
                for j,yolo in enumerate(model.yolo_layers):         # yolo_layer有metrics属性
                    for name,metric in yolo.metrics.items():        # items()以列表返回可遍历的(键, 值) 元组数组
                        if name != "grid_size":                     # 如果metric的名称不是grid_size就让tensorboard_log记录下来
                            tensorboard_log +=[("{}_{}".format(name,j+1),metric)]
                tensorboard_log += [("loss",loss.item())]

            log_str += "\nTotal loss {}".format(loss.item())      # log日志的str

            # Determine approximate time left for epoch.确定epoch大概剩余的时间
            epoch_batches_left = len(dataLoader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str +="\n ----- ETA{}".format(time_left)
            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval  == 0:
            print("\n---- Evaluating Model ----")
            # 每隔opt.evaluation_interval个间隔就评价
            # 下面都是评价指标,（精确率、召回率
            precision,recall,AP,f1,ap_class = evaluate(
                model,
                root = opt.root,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )
            evaluation_metrics = [
                ("val_precision",precision.mean()),
                ("val_recall",recall.mean()),
                ("val_mAP",AP.mean()),
                ("val_f1",f1.mean()),
            ]

            # Print class APs and mAP
            ap_table = [["Index","Class name","AP"]]
            for i,c in enumerate(ap_class):
                ap_table +=[[c,class_name[c],"%.5f"%AP[i]]]
            print("-----mAP {}".format(AP.mean()))


        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(),"weights/yolov3_ckpt_%d.pth" % epoch)

























