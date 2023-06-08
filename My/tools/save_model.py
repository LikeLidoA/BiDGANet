import os
import torch


def save_checkpoint(model, cfg, epoch=0, is_best=False, mIoU=0.0, dataparallel=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg["train"]["ckpt_dir"])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}_{:.3f}.pth'.format(cfg["model"]["name"], cfg["model"]["backbone"], epoch, mIoU)
    filename = os.path.join(directory, filename)
    if dataparallel:
        model = model.module
    if is_best:
        best_filename = '{}_{}_{}_{:.3f}_best_model.pth'.format(cfg["model"]["name"], cfg["model"]["backbone"], epoch,
                                                                mIoU)
        best_filename = os.path.join(directory, best_filename)
        torch.save(model.state_dict(), best_filename)
