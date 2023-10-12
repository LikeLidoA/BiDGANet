import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# 定义Grad-CAM类
class GradCAM:
    def __init__(self, model, target_layer):
        self.handles = None
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]

        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        target_layer = self.model._modules.get(self.target_layer)
        backward_hook_handle = target_layer.register_backward_hook(backward_hook)
        forward_hook_handle = target_layer.register_forward_hook(forward_hook)

        self.handles = [backward_hook_handle, forward_hook_handle]

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def generate_cam(self, input_image, class_index=None):
        input_image.requires_grad = True

        model_output = self.model(input_image)
        if class_index is None:
            class_index = torch.argmax(model_output)

        self.model.zero_grad()
        one_hot = torch.zeros((1, model_output.size()[-1]), dtype=torch.float32)
        one_hot[0][class_index] = 1
        model_output.backward(gradient=one_hot, retain_graph=True)

        gradient_maps = self.gradient.mean(dim=[2, 3], keepdim=True)
        cam = (gradient_maps * self.feature_maps).sum(dim=1, keepdim=True)
        cam = nn.functional.relu(cam)

        cam = nn.functional.interpolate(cam, input_image.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


if __name__ == '__main__':
    # 加载ResNet18模型
    model = resnet18(pretrained=True)
    model.eval()

    # 加载示例图像
    image_path = 'example_image.jpg'
    input_image = Image.open(image_path)

    # 预处理图像
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)

    # 生成Grad-CAM
    cam_generator = GradCAM(model, target_layer='layer4')
    cam = cam_generator.generate_cam(input_tensor)

    # 可视化Grad-CAM
    plt.imshow(input_image)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

    # 移除hook
    cam_generator.remove_hooks()
