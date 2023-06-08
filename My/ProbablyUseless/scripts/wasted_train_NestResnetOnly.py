#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import torch
import os.path as osp
import logging
import argparse
# import apex

from ProbablyUseless.tools.cityscapes_cv2 import get_data_loader
from models.Nest_ResNet1 import ReHalf_U2NET
from tools.ohem_ce_loss import OhemCELoss
from ProbablyUseless.tools.meters import TimeMeter, AvgMeter
from ProbablyUseless.tools.logger import setup_logger, print_log_msg
from tools.lr_scheduler import WarmupPolyLrScheduler
from ProbablyUseless.tools.evaluatev2 import eval_model


has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False


# def set_syncbn(net):
#     if has_apex:
#         net = parallel.convert_syncbn_model(net)
#     else:
#         net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
#     return net

# def set_model_dist(net):
#     if has_apex:
#         net = parallel.DistributedDataParallel(net, delay_allreduce=True)
#     else:
#         local_rank = dist.get_rank()
#         net = nn.parallel.DistributedDataParallel(
#             net,
#             device_ids=[local_rank, ],
#             output_device=local_rank)
#     return net

def parse_args():  # 此处可以添加运行参数
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_type', dest='model_type', type=str, default='ReHalf_U2NET', )
    parse.add_argument('--result_pth', dest='result_pth', type=str, default='./results', )
    return parse.parse_args()


args = parse_args()
lr_start = 5e-2
warmup_iters = 1000
max_iter = 150000 + warmup_iters
ims_per_gpu = 2


def save_model(states, save_pth):
    logger = logging.getLogger()
    logger.info('\nsave models to {}'.format(save_pth))
    for name, state in states.items():
        save_name = 'model_final_{}.pth'.format(name)
        modelpth = osp.join(save_pth, save_name)
        # if dist.is_initialized() and dist.get_rank() == 0:
        torch.save(state, modelpth)


def set_meters():
    time_meter = TimeMeter(max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    # loss_aux_meters = [AvgMeter('loss_aux{}'.format(i)) for i in range(4)]
    return time_meter, loss_meter, loss_pre_meter  # , loss_aux_meters


def set_optimizer(model):
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if param.dim() == 1:
            non_wd_params.append(param)
        elif param.dim() == 2 or param.dim() == 4:
            wd_params.append(param)
    params_list = [
        {'params': wd_params, },
        {'params': non_wd_params, 'weight_decay': 0},
    ]
    optim = torch.optim.SGD(
        params_list,
        lr=lr_start,
        momentum=0.9,
        weight_decay=5e-4,
    )
    return optim


def set_model():
    net = ReHalf_U2NET(19)
    # if args.use_sync_bn: net = set_syncbn(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    # criteria_aux = [OhemCELoss(0.7) for _ in range(4)]
    return net, criteria_pre  # , criteria_aux


def train():
    logger = logging.getLogger()
    # is_dist = dist.is_initialized()

    # dataset
    datalist = get_data_loader('E:\\B\cityscape\\leftImg8bit_trainvaltest', ims_per_gpu, max_iter, mode='train')

    # model
    net, criteria_pre = set_model()

    # optimizer
    optim = set_optimizer(net)

    # fp16
    # if has_apex and args.use_fp16:
    #     net, optim = amp.initialize(net, optim, opt_level='O1')

    # ddp training
    # net = set_model_dist(net)

    # meters
    # time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()
    time_meter, loss_meter, loss_pre_meter = set_meters()

    # lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
                                     max_iter=max_iter, warmup_iter=warmup_iters,
                                     warmup_ratio=0.1, warmup='exp', last_epoch=-1, )

    # train loop
    for it, (image_data, lossbackward) in enumerate(datalist):
        image_data = image_data.cuda()
        lossbackward = lossbackward.cuda()
        lossbackward = torch.squeeze(lossbackward, 1)  # torch.squeeze函数的作用是对输入的张量进行处理，如果张量维度里面有大小为1 的部分，那我们就移除，否则保留
        optim.zero_grad()
        logits = net(image_data)  # 全连接层输出
        loss_pre = criteria_pre(logits, lossbackward)
        loss = loss_pre
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
        lr_schdr.step()
        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        # print training log message
        if (it + 1) % 100 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(it, max_iter, lr, time_meter, loss_meter,
                          loss_pre_meter, loss_aux_meters=None)
    # dump the final model and evaluate the result
    save_pth = osp.join(args.result_pth, 'model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.module.state_dict()
    torch.save(state, save_pth)

    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    eval_model(net, 4)


def main():
    torch.cuda.set_device(0)
    # dist.init_process_group(
    #     backend='nccl',
    #     init_method='tcp://127.0.0.1:{}'.format(args.port),
    #     world_size=torch.cuda.device_count(),
    #     rank=args.local_rank
    # )
    # if not osp.exists(args.respth): os.makedirs(args.respth)
    setup_logger('ReHalf_U2NET-train', args.result_pth)
    train()


if __name__ == "__main__":
    main()
