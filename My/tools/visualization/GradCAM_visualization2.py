import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F


class ResNet18(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def get_layer_feature_map(model, x, layer_name):
    """获取输入x在指定名称的卷积层的feature map"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name == layer_name:
            return x
        if isinstance(module, nn.Conv2d):
            x = module(x)


def get_gradcam(model, x, class_idx, layer_name):
    """计算指定卷积层的Grad-CAM"""
    features = get_layer_feature_map(model, x, layer_name)
    _, c, h, w = features.size()
    score = model(x)
    y = score[0, class_idx]
    y.backward()
    weight = F.adaptive_avg_pool2d(model.lastlayer.weight, (1, 1))
    gradcam = torch.mul(weight, features)
    gradcam = gradcam.sum(dim=1, keepdim=True)
    gradcam = F.relu(gradcam)
    gradcam = F.interpolate(gradcam, (h, w), mode="bilinear", align_corners=False)
    gradcam = gradcam.squeeze(0)
    gradcam = gradcam.cpu().numpy()
    gradcam = np.maximum(gradcam, 0)
    gradcam = gradcam / gradcam.max()
    return gradcam


# 加载模型和图像
model = ResNet18()
model.load_state_dict(torch.load("resnet18.pth"))
model.eval()

img = Image.open("test_image.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
img_tensor = transform(img).unsqueeze(0)

# 计算Grad-CAM
class_idx = 1
layer_name = "layer3.1.conv2"  # 设置要可视化的卷积层
gradcam = get_gradcam(model, img_tensor, class_idx, layer_name)

# 可视化Grad-CAM
img_np = np.array(img)
plt.imshow(img_np)
plt.imshow(gradcam, cmap="jet", alpha=0.5)
plt.axis('off')
plt.show()
