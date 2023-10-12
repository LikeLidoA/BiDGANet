import os
import time
import datetime
import yaml
import shutil
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data

from PIL import Image
from torchvision import transforms
# from models import Nest_ResNet1
from models import Nest_ResNet2
from models import networks
from datasets.cityscapes import get_city_pairs
from tools.logger import SetupLogger
from tools.metric import SegmentationMetric
from tools.visualize import get_color_pallete


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return param_size, param_sum, buffer_size, buffer_sum, all_size


def _img_transform(image):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    image = image_transform(image)
    return image


class Evaluator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化验证集的图片和标签，或是初始化test集
        self.image_paths, self.mask_paths = get_city_pairs(cfg["train"]["cityscapes_root"], "val")

        # 模型初始化
        self.model = networks.Mymodel().to(self.device)

        # 载入权重
        pretrained_net = torch.load(cfg["test"]["ckpt_path"])
        self.model.load_state_dict(pretrained_net)

        # 评价指标的初始化
        self.metric = SegmentationMetric(19)

        self.current_mIoU = 0

    def eval(self):
        self.metric.reset()
        self.model.eval()
        model = self.model

        logger.info("开始验证集的评估, 样本总计: {:d}".format(len(self.image_paths)))
        list_time = []
        lsit_pixAcc = []
        list_mIoU = []

        for i in range(len(self.image_paths)):
            image = Image.open(self.image_paths[i]).convert('RGB')  # image shape: (W,H,3)
            mask = Image.open(self.mask_paths[i])  # mask shape: (W,H)

            image = _img_transform(image)  # image shape: (3,H,W) [0,1]
            mask = self._mask_transform(mask)  # mask shape: (H,w)

            image = image.to(self.device)
            mask = mask.to(self.device)
            image = torch.unsqueeze(image, 0)  # image shape: (1,3,H,W) [0,1]

            with torch.no_grad():
                start_time = time.time()
                outputs = model(image)
                end_time = time.time()
                step_time = end_time - start_time
            self.metric.update(outputs, mask)
            pixAcc, mIoU = self.metric.get()
            list_time.append(step_time)
            lsit_pixAcc.append(pixAcc)
            list_mIoU.append(mIoU)
            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}, time: {:.3f}s".format(
                i + 1, pixAcc * 100, mIoU * 100, step_time))

            filename = os.path.basename(self.image_paths[i])
            prefix = filename.split('.')[0]

            # save target
            mask = Image.open(self.mask_paths[i])  # mask shape: (W,H)
            mask = np.array(mask)
            mask = self._class_to_index(mask).astype('int32')
            mask_temp = (mask != -1).astype(int)
            mask = get_color_pallete(mask, "citys")
            mask.save(os.path.join(outdir, prefix + '_label.png'))

            # save pred
            pred = torch.argmax(outputs, 1)+1
            pred = pred.cpu().data.numpy()
            pred = pred.squeeze(0)
            pred = pred*mask_temp-1
            pred = get_color_pallete(pred, "citys")
            pred.save(os.path.join(outdir, prefix + "_mIoU_{:.3f}.png".format(mIoU)))

            # save image
            image = Image.open(self.image_paths[i]).convert('RGB')  # image shape: (W,H,3)
            image.save(os.path.join(outdir, prefix + '_src.png'))

        average_pixAcc = sum(lsit_pixAcc) / len(lsit_pixAcc)
        average_mIoU = sum(list_mIoU) / len(list_mIoU)
        average_time = sum(list_time) / len(list_time)
        self.current_mIoU = average_mIoU
        logger.info(
            "Evaluate: Average mIoU: {:.3f}, Average pixAcc: {:.3f}, Average time: {:.3f}".format(average_mIoU,
                                                                                                  average_pixAcc,
                                                                                                  average_time))
        getModelSize(self.model)

    def _mask_transform(self, mask):
        mask = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')
        for value in values:
            assert (value in self._mapping)
        # 获取mask中各像素值对应于_mapping的索引
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        # 依据上述索引index，根据_key，得到对应的mask图
        return self._key[index].reshape(mask.shape)


if __name__ == '__main__':
    # 设置参数表
    config_path = "./configs/ReHalf_U2Net_city.yaml"
    with open(config_path, "r", encoding='utf-8') as yaml_file:
        cfg = yaml.safe_load(yaml_file.read())
        # print(cfg)
        # print(cfg["model"]["backbone"])
        print(cfg["train"]["specific_gpu_num"])

    # 指定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"]["specific_gpu_num"])
    num_gpus = len(cfg["train"]["specific_gpu_num"].split(','))
    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
    print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    print("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))

    outdir = os.path.join(cfg["train"]["ckpt_dir"], "evaluate_output")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logger = SetupLogger(name="semantic_segmentation",
                         save_dir=cfg["train"]["ckpt_dir"],
                         distributed_rank=0,
                         filename='{}_{}_evaluate_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))

    evaluator = Evaluator(cfg)
    evaluator.eval()
