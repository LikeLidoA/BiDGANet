import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from models.Nest_ResNet1 import ReHalf_U2NET
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms
from models import networks


def feature_vis(feats):  # features形状: [b,c,h,w]
    output_shape = (1024, 2048)  # 输出形状
    feats = F.interpolate(feats, size=output_shape, mode='bilinear', align_corners=False)
    feats = feats.squeeze(0).squeeze(0).cpu().detach().numpy()
    x_visualize = np.max(feats, axis=0)
    x_visualize = (((x_visualize - np.min(x_visualize)) / (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(
        np.uint8)

    savedir = '../../results/visualization/'
    # if not os.path.exists(savedir + 'results\\visualization'): os.makedirs(savedir + 'feature_vis')
    channel_mean = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)
    #cv2.namedWindow("image")  # 创建一个image的窗口
    cv2.imshow("image", channel_mean)  # 显示图像
    cv2.waitKey()  # 默认为0，无限等待
    cv2.destroyAllWindows()  # 释放所有窗口
    # cv2.imwrite(savedir + 'feature_without_head_channel_max.png', channel_mean)
    cv2.imwrite(savedir + 'S3_channel_max.png', channel_mean)


# def feature_vis(feats):  # features形状: [b,c,h,w]
#     output_shape = (1024, 2048)  # 输出形状
#     channel_mean = torch.mean(feats, dim=1, keepdim=True)  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
#     channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
#     channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().detach().numpy()  # 四维压缩为二维
#     channel_mean = (
#             ((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255).astype(
#         np.uint8)
#     savedir = './results/visualization/'
#     # if not os.path.exists(savedir + 'results\\visualization'): os.makedirs(savedir + 'feature_vis')
#     channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
#     # cv2.namedWindow("image")  # 创建一个image的窗口
#     # cv2.imshow("image", channel_mean)  # 显示图像
#     # cv2.waitKey()  # 默认为0，无限等待
#     # cv2.destroyAllWindows()  # 释放所有窗口
#     cv2.imwrite(savedir + 'feature_without_head_channel_mean.png', channel_mean)


if __name__ == '__main__':
    savedir = '../../results/visualization/'
    # model = ReHalf_U2NET(19)
    model = networks.HighResolutionBranch()
    # model.load_state_dict(torch.load('./results/ckpt/ReHalf_U2NET_without-backbone_888_0.718_best_model.pth'),strict=True)
    model.load_state_dict(torch.load('../../results/ckpt/HighResolutionBranch_96_0.335_best_model.pth'),strict=True)
    model.eval()
    size_upsample = (1024, 2048)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # 对于cityscapes数据集标准化
    # normalize = transforms.Normalize(
    #     mean=[0.2860406, 0.32444712, 0.2832653],
    #     std=[0.17639251, 0.17330144, 0.17715833])

    preprocess = transforms.Compose([
        transforms.Resize(size_upsample),
        transforms.ToTensor(),
        normalize])
    image_path = '../../bochum_000000_023435_leftImg8bit.png'
    img_pil = Image.open(image_path)
    img = cv2.imread(image_path)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    output= model(img_variable)
    # output, stage1, stage2, stage3, stage4, stage5, feature_without_head = model(img_variable)
    # output, stage1 = model(img_variable)
    feature_vis(output)
    # feature_vis(stage1)
    # feature_vis(stage2)
    # feature_vis(stage3)
    # feature_vis(stage4)
    # feature_vis(stage5)
    # feature_vis(feature_without_head)
