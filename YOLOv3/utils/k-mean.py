"""
    通过k-mean聚类获得anchor boxes的width和height。即从已标注的数据中通过聚类统计到的最有可能的object的形状.
    k-mean算法：（1）随便指定k个clustser，并把点划分到与之最近的一个cluster。这时的cludster肯定是不好的，因为一开始是乱选的。
               （2）更新每个cluster为当前cluster的点的均值；
               （3）重复执行上述过程，把点划分到与之最近的一个cluster；更新每个cluster为当前cluster的点的均值。

    一般来说，我们都是用计算样本点到质点的距离大小来判断属于哪个质点，但是在yolo v3中并不是直接结算两点之间的距离，而是计算两个box的iou，即两个box的相似程度。d=1-iou(box1,cox_cluster)，
    当d越小，说明box1和box_cluster越类似，将box1划归为box_cluster.
"""
from os import listdir
from os.path import isfile, join
import argparse
import cv2 as cv
import numpy as np
import sys
import os
import shutil
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

width_in_cfg_file = 416     # 网络输入大小
height_in_cfg_file =416     # 网络输入大小

# 定义平均iou函数
def avg_iou(x,centroids):
    """
        Args:
            x: 所有样本的w和h
          centroids:质心w和h
    """
    n,d = x.shape       # n代表样本个数，d代表样本的深度，这里是(w,h)
    sum =  0
    for i in range(x.shape[0]):
        # 遍历所有的样本
        sum += max(iou(x[i],centroids))     # 求出所有样本的iou总和
    return sum/n


# 将数据写入到txt文档保存
def write_anchors_to_file(centroids,X,anchor_file):
    """
    Args:
        centroids:质心的h和w
        X: 样本数量
        anchor_file:输出的文件路径
    """
    f = open(anchor_file,'w')       # 打开文件

    anchors = centroids.copy()      # anchors保存质心的(h,w)
    print("anchors.shape",anchors.shape)

    for i in range(anchors.shape[0]):   # 遍历每个质心
        # 因为yolo中的label是width、height的比例，所以得到的anchor box的大小要乘以模型输入图片的尺寸
        anchors[i][0] *= width_in_cfg_file
        anchors[i][1] *= height_in_cfg_file

    widths = anchors[:,0]       # 获取每个质心的w，这里的w是乘以图片的尺寸后的
    sorted_indices = np.argsort(widths)     # 将width排序并返回排序后索引下标

    print('Anchors=',anchors[sorted_indices])       # anchors[sorted_indices]:代表返回按w排序后的anchors

    for i in sorted_indices[:-1]:       # 遍历所有的w
        f.write('w: %0.2f , h: %0.2f\n, ' % (anchors[i, 0], anchors[i, 1]))       # 写下每个质心的w和h

    # 最后一个锚之后不应该有逗号，这就是为什么
    f.write('w: %0.2f , h: %0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))
    f.write('arg_iou:%f\n' % (avg_iou(X, centroids)))       # 写下avg_iou
    print()







# 定义iou,求每个样本和质心的iou
def iou(x,centroids):
    similarities = []   # 用来保存相似程度
    k = len(centroids)    # 用来遍历所有的质心，长度

    for centroid in centroids:
        c_w,c_h = centroid      # 获取每一个质心的w和h
        w,h = x                 # 样本的w和h

        # 当质心的w和h大于或等于样本的w和h时，iou就等于下式
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # 意味着w，h分别大于c_w和c_h
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # 将变成（k，）形状,返回的similarities.shape是(20,)

    return np.array(similarities)




# 定义k-mean算法
def kmeans(X,centroids,eps,anchor_file):
    """
    Args:
        X: 15762个（w，h）组成的列表,  X.shape = N * dim,N代表全部样本数量，dim表示样本有dim个维度
        centroids: 从15762中随机选择k个（w，h）组成质心，  centroids.shape=k*dim,k代表聚类的cluster数，dim代表样本维度
        eps:最小调整幅度阈，调整幅度小于阈值，则停止运行。
        anchor_file:输出的文件路径
    :return:
    """
    print("X.shape=",X.shape,"cetroids.shape=",centroids.shape)

    N = X.shape[0]      # N代表全部的样本数量
    iterations = 0
    k,dim = centroids.shape     # 质心的cluster数（k是质心的数量），和样本维度（这里是2维）
    prev_assignments = np.ones(N)*(-1)  # 用来和计算出的距离进行比较，值为-1
    iter = 0
    old_D = np.zeros((N,k))     # N是全部样本，k是质心数量，old_D表示全部样本到k个质心的距离,这里是全为0

    while True:
        """
            D.shape=N * k ,N代表全部样本数量，k列分别为到k个质心的距离
            1、计算出D
            2、获取出当前样本应该归属于哪个cluster
            assignments = np.argmin(D,axis=1)
            assignments.shape = N * 1，N代表N个样本，1列为当前归属哪个cluster
            numpy里row=0，line=1，np.argmin(D,axis=1)即沿着列的方向，即每一行的最小值的下标
            3、将样本划分到相对应的cluster后，重新计算每个cluster的质心
                centroid_sums.shape = k * dim,K代表几个cluster，dim列分别为该cluster的样本在该维度的均值。
                
            centroid_sums = np.zeros((k,dim),np.float)
            for i in range(N):
                centroid_sums[assignments[i]]+= X[i]        # assignments[i]为cluster x，将每一个样本都归到其所属的cluster
            for j in range(k):
                centroids[j] = centroid_sums[j] / (np.sum(assignment == j)) # np.sum(assignment == j)为cluster j中的样本总量
        """
        D = []
        iter +=1
        for i in range(N):  # 遍历全部样本数
            d = 1 - iou(X[i],centroids)     # 求每个样本和所有质心的距离（iou）,iou()返回的是shape=(20,)
            D.append(d)                     # d的shape是（20，），D是添加了i（也就是N）个维度为（20，）的数据
        D = np.array(D)     # D.shape = (N,k),有N个样本和k个质心的iou
        print('D.shape',D.shape)
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D - D))))  # old_D - D :表示0-D的绝对值，代表样本和质心的距离
        assignments = np.argmin(D,axis=1)   # np.argmin():寻找最小值，并返回下标，axis=1代表找到d值最小的哪个质心的下标,从20个质心中选1个和每个样本距离最小的，assignment.shape(15672,)
        print("assignments",assignments)      # 将每一个样本和距离最近的cluster的下标返回给assignments，assigments=[3,2,1,5,6,1,3,...]共15672个

        # 每个样本归属的cluster都不再变化了，就退出
        if( assignments == prev_assignments).all():
            print("centroids = ",centroids)
            write_anchors_to_file(centroids,X,anchor_file)
            return

        # 计算新的质点
        centroids_sum = np.zeros((k,dim),np.float32)      # 新质点的shape=(k,dim),k是质点个数，dim代表(w,h) 2个维度
        for i in range(N):          # centroids_sum[3]代表第3个质点的(w,h) + 每一个样本(w,h)
            centroids_sum[assignments[i]] += X[i]   # assignment[i]返回每个样本和最小距离的那个质心的（w，h）的索引，即centroids_sum[assignment[i]]:返回的是每个样本和最小距离的质心的（w,h)

        for j in range(k):
            print('cluster{} has {} sample'.format(j,np.sum(assignments)))  # 第j个cluster具体有多少个
            centroids[j] = centroids_sum[j] / (np.sum(assignments == j))    #

        prev_assignments = assignments.copy()
        old_D = D.copy()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist",default=r'E:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\train.txt',help='文件列表的路径')      # 训练文件的路径
    parser.add_argument("--output_dir",default=r'E:\dataset\voc2yolo',type=str,help='输出anchor的目录')      # 输出anchors的目录
    parser.add_argument('--num_clusters',default=9,type=int,help='要分成几类')       # 需要预测锚框个数
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        # 如果输出文件目录不存在，就新创建一个
        os.makedirs(args.output_dir)

    # 打开文件列表
    f = open(args.filelist)
    # 获取train.txt里面的图像名称前缀
    lines = [line.rstrip('\n') for line in f.readlines()]   # train.txt的路径

    # 将label文件里的obj的w_ratio,h_ratio存储到annotation_dims中
    annotation_dims = []
    for line in lines:      # 遍历train.txt中所有的文件名称，更换成label的路径
        img_path = line     # 保存图片的路径
        #print(img_path)

        line = line.replace('JPEGImages', 'labels')     # 更换路径
        line = line.replace('.jpg', '.txt')             # 更换后缀
        line = line.replace('.png', '.txt')

        f2 = open(line)        # 打开label文件
        for line in f2.readlines():     # 遍历label中的txt文件中的每一行
            line = line.rstrip('\n')
            w,h = line.split(' ')[3:]    # 获取txt文件中的高和宽
            # print(w,h)
            annotation_dims.append(tuple(map(float,(w,h))))     # map() 会根据提供的函数对指定序列做映射。map(function, iterable, ...),第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
            # print(annotation_dims)                              # 生成一个N*2的矩阵，N代表样本个数

    annotation_dims = np.array(annotation_dims)
    print(annotation_dims.shape[0])

    eps = 0.005

    # # 如果输入的分类数等于0
    if args.num_clusters == 0:
        for num_clusters in range(1,11):    # 我们就制作1到10个集群
            anchor_file = join(args.output_dir,'anchors%d.txt'%(num_clusters))  # anchors输出的txt路径

            indices = [random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]  # 随机选择,annotation_dims.shape=(15762，2）,而annotation_dims.shape[0]=15762
            print('indices',indices)
            centroids = annotation_dims[indices]        # 从15762个(w,h)中中随机选择k(这里是10)个质心，也就是随机选择(w,h)
            print("centroids.shape",centroids.shape)
            kmeans(annotation_dims,centroids,eps,anchor_file)

    else:
        anchor_file = join(args.output_dir,"anchors%d.txt"%(args.num_clusters))

        # 随机选取args.num_clusters个质心
        indices = [ random.randrange(annotation_dims.shape[0]) for i in range(args.num_clusters)]   # indices随机选择20个索引
        print("indeces={}".format(indices))
        centroids = annotation_dims[indices]    # 根据索引获得20个质心的w，h，质心的(w,h),这里总共20个质心，每个质心都是(w,h)组成的
        print("centroids= ",centroids)

        # k-mean
        kmeans(annotation_dims,centroids,eps,anchor_file)
        print('centroids.shape',centroids.shape)

        




