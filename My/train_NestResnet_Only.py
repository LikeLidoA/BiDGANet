# -*- coding: UTF-8 -*-
import os
import torch
import time
import datetime
import yaml
import torch.utils.data as data
from tools.metric import SegmentationMetric
from tools.logger import SetupLogger
from datasets.cityscapes import CityscapesDataset
from models.Nest_ResNet2 import ReHalf_U2NET
from tools.ohem_ce_loss import OhemCELoss
from tools.lr_scheduler import WarmupPolyLrScheduler
from tools.save_model import save_checkpoint
from models.networks import LowResolutionModel_Only


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_parallel = torch.cuda.device_count() > 1

        # dataset and dataloader
        train_dataset = CityscapesDataset(root=cfg["train"]["cityscapes_root"],
                                          split='train',
                                          base_size=cfg["model"]["base_size"],
                                          crop_size=cfg["model"]["crop_size"])
        val_dataset = CityscapesDataset(root=cfg["train"]["cityscapes_root"],
                                        split='val',
                                        base_size=cfg["model"]["base_size"],
                                        crop_size=cfg["model"]["crop_size"])

        self.train_dataloader = data.DataLoader(dataset=train_dataset,
                                                batch_size=cfg["train"]["train_batch_size"],
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True,
                                                drop_last=False)
        self.val_dataloader = data.DataLoader(dataset=val_dataset,
                                              batch_size=cfg["train"]["valid_batch_size"],
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              drop_last=False)

        self.iters_per_epoch = len(self.train_dataloader)
        self.warmup_iters = 1000
        self.max_iters = cfg["train"]["epochs"] * self.iters_per_epoch + self.warmup_iters

        # model初始化
        # self.net = ReHalf_U2NET(train_dataset.NUM_CLASS).cuda(self.device)
        self.net = LowResolutionModel_Only(train_dataset.NUM_CLASS).cuda(self.device)

        # optimizer初始化，此处还可以添加模型自带参数，如预训练模型参数等
        wd_params = []
        for name, param in self.net.named_parameters():
            wd_params.append(param)
        params_list = [
            {'params': wd_params, }
        ]
        self.optimizer = torch.optim.SGD(params=params_list,
                                         lr=cfg["optimizer"]["init_lr"],
                                         momentum=cfg["optimizer"]["momentum"],
                                         weight_decay=cfg["optimizer"]["weight_decay"])

        # lr scheduler学习率的策略
        self.lr_sch = WarmupPolyLrScheduler(self.optimizer, power=0.9,
                                            max_iter=self.max_iters, warmup_iter=self.warmup_iters,
                                            warmup_ratio=0.1, warmup='exp', last_epoch=-1, )
        # 损失函数初始化
        self.criteria_pre = OhemCELoss(0.7).to(self.device)
        # 评价指标初始化
        self.metric = SegmentationMetric(train_dataset.NUM_CLASS)

        # 其他小参数
        self.current_mIoU = 0.0
        self.best_mIoU = 0.0
        self.epochs = cfg["train"]["epochs"]
        self.current_epoch = 0
        self.current_iteration = 0

    def train(self):
        epochs, max_iters = self.epochs, self.max_iters
        log_per_iters = self.cfg["train"]["log_iter"]
        val_per_iters = self.cfg["train"]["val_epoch"] * self.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.net.train()

        for _ in range(self.epochs):
            torch.cuda.empty_cache()
            self.current_epoch += 1
            list_pixAcc = []
            list_mIoU = []
            list_loss = []
            self.metric.reset()
            for i, (images, targets, _) in enumerate(self.train_dataloader):
                # 准备学习率
                self.current_iteration += 1
                # self.lr_sch.step()

                # 图片的载入
                images = images.to(self.device)
                targets = targets.to(self.device)

                # 图片数据从网络中输出
                outputs = self.net(images)

                loss = self.criteria_pre(outputs, targets)

                # 计算评价指标
                self.metric.update(outputs, targets)
                pixAcc, mIoU = self.metric.get()
                list_pixAcc.append(pixAcc)
                list_mIoU.append(mIoU)
                list_loss.append(loss.item())

                # 反向传播更新
                self.optimizer.zero_grad()
                loss.backward()
                # torch.autograd.backward(loss)
                self.optimizer.step()

                # if (i + 1) % accumulation_steps == 0:
                #     optimizer.step()
                #     optimizer.zero_grad()

                # 记录时间
                torch.cuda.synchronize(self.device)
                eta_seconds = ((time.time() - start_time) / self.current_iteration) * (
                        max_iters - self.current_iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.lr_sch.step()

                if self.current_iteration % log_per_iters == 0:
                    logger.info(
                        "Epochs: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                            self.current_epoch, self.epochs,
                            self.current_iteration, max_iters,
                            self.optimizer.param_groups[0]['lr'],
                            loss.item(),
                            mIoU,
                            str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            # 计算平均指标
            average_pixAcc = sum(list_pixAcc) / len(list_pixAcc)
            average_mIoU = sum(list_mIoU) / len(list_mIoU)
            average_loss = sum(list_loss) / len(list_loss)
            logger.info(
                "Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(
                    self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))

            # 评估模型再训练
            if self.current_iteration % val_per_iters == 0:
                self.validation()
                self.net.train()

        # 计算此epoch的时间开销
        torch.cuda.synchronize(self.device)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self):
        is_best = False
        self.metric.reset()
        # if self.dataparallel:
        #     model = self.model.module
        # else:
        #     model = self.model
        model = self.net
        model.eval()
        list_pixAcc = []
        list_mIoU = []
        list_loss = []
        torch.cuda.empty_cache()
        for i, (image, targets, filename) in enumerate(self.val_dataloader):
            image = image.to(self.device)
            targets = targets.to(self.device)
            with torch.no_grad():
                outputs = model(image)
                loss = self.criteria_pre(outputs, targets)

            self.metric.update(outputs, targets)
            pixAcc, mIoU = self.metric.get()
            list_pixAcc.append(pixAcc)
            list_mIoU.append(mIoU)
            list_loss.append(loss.item())

        average_pixAcc = sum(list_pixAcc) / len(list_pixAcc)
        average_mIoU = sum(list_mIoU) / len(list_mIoU)
        average_loss = sum(list_loss) / len(list_loss)
        self.current_mIoU = average_mIoU
        logger.info(
            "Validation: Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(average_loss,
                                                                                                    average_mIoU,
                                                                                                    average_pixAcc))
        if self.current_mIoU > self.best_mIoU:
            is_best = True
            self.best_mIoU = self.current_mIoU
        if is_best:
            save_checkpoint(self.net, self.cfg, self.current_epoch, is_best, self.current_mIoU, self.data_parallel)


if __name__ == '__main__':
    # Set config file
    config_path = "./configs/ReHalf_U2Net_city.yaml"
    with open(config_path, "r", encoding='utf-8') as yaml_file:
        cfg = yaml.safe_load(yaml_file.read())
        # print(cfg)
        # print(cfg["model"]["backbone"])
        # print(cfg["train"]["specific_gpu_num"])

    # Use specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"]["specific_gpu_num"])
    num_gpus = len(cfg["train"]["specific_gpu_num"].split(','))
    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
    print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    print("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))

    # Set logger
    logger = SetupLogger(name="semantic_segmentation",
                         save_dir=cfg["train"]["log_save_dir"],
                         distributed_rank=0,
                         filename='{}_{}_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
    logger.info("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    logger.info("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))
    logger.info(cfg)

    # Start train
    trainer = Trainer(cfg)
    trainer.train()
