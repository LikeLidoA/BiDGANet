import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载数据集
data = datasets.ImageFolder(root='path/to/dataset', transform=transform)

# 加载数据
loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

# 统计均值和标准差
mean = 0.
std = 0.
total = 0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total += batch_samples

mean /= total
std /= total

print('mean:', mean)
print('std:', std)
