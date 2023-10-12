import torch
# from torchstat import stat
from torch import nn
from models import Nest_ResNet2
from models.blocks import ConvTransposeBnReLUDeep, CARAFE
from models import blocks
import torchvision.transforms as transforms
from PIL import Image
import argparse
# import cv2
import torch

# from torchsummary import summary

if __name__ == '__main__':
    x = torch.randn(1, 64, 128, 256)
    CARAFE = CARAFE(64)
    x = CARAFE(x)
    # TranConv = ConvTransposeBnReLUDeep(64, 128)
    # x=TranConv(x)
    print(x)
    # x = torch.linspace(1, 90, steps=90).view(2, 3, 3, 5)
    # print(x)
    # x = x.view(2, 3, 15)
    # print(x)
    # x = x.permute(0, 2, 1)
    # print(x)
    # x = x.permute(0, 2, 1)
    # print(x)
    # x = x.view(2, 3, 3,5)
    # print(x)
    # device = torch.device('cpu')
    # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Nest_ResNet2.ReHalf_U2NET(19).to(device)
    # # x = torch.randn(3, 1024, 2048)
    # stat(model, (3, 2048, 1024))
    # # summary(model, (3, 1024, 512))
    # print(stat)
    # # print(summary)
    # #
    # # model_weights = []  # append模型的权重
    # # conv_layers = []  # append模型的卷积层本身
    # #
    # # # get all the model children as list
    # # model_children = list(model.children())
    # # # k = type(model_children[0])
    # # print(model_children)
