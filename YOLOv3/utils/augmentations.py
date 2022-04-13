"""
    此文件是用来增强图片的
"""
import torch
import torch.nn.functional as F
import numpy as np

def horisontal_flip(images,targets=None):
    # 水平翻转
    images = torch.flip(images,dims=[2])
    # 当dim=[1]是上下翻转
    #print(images)
    return images,targets

if __name__=='__main__':
    a = torch.randn((2, 3, 4))
    #print(a)
    #print(a.shape)
    img,target= horisontal_flip(a)
    #print(img.shape)