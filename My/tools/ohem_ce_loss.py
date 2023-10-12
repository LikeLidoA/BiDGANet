#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


#
# class OhemCrossEntropyLoss(nn.Module):
#     """
#     Implements the ohem cross entropy loss function.
#     Args:
#         thresh (float, optional): The threshold of ohem. Default: 0.7.
#         min_kept (int, optional): The min number to keep in loss computation. Default: 10000.
#         ignore_index (int64, optional): Specifies a target value that is ignored
#             and does not contribute to the input gradient. Default ``255``.
#     """
#
#     def __init__(self, thresh=0.7, min_kept=10000, ignore_index=255):
#         super(OhemCrossEntropyLoss, self).__init__()
#         self.thresh = thresh
#         self.min_kept = min_kept
#         self.ignore_index = ignore_index
#         self.EPS = 1e-5
#
#     def forward(self, logit, label):
#         """
#         Forward computation.
#         Args:
#             logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
#                 (N, C), where C is number of classes, and if shape is more than 2D, this
#                 is (N, C, D1, D2,..., Dk), k >= 1.
#             label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
#                 value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
#                 (N, D1, D2,..., Dk), k >= 1.
#         """
#         if len(label.shape) != len(logit.shape):
#             label = torch.unsqueeze(label, 1)
#
#         # get the label after ohem
#         n, c, h, w = logit.shape
#         label = label.view((-1,))
#         valid_mask = torch.tensor((label != self.ignore_index), dtype=torch.int64)  # .astype('int64')
#         num_valid = valid_mask.sum()
#         label = label * valid_mask
#
#         prob = F.softmax(logit, dim=1)
#         prob = prob.transpose(0, 1).view((c, -1))
#         # prob = prob.transpose((1, 0, 2, 3)).reshape((c, -1))
#
#         if self.min_kept < num_valid and num_valid > 0:
#             # let the value which ignored greater than 1
#             prob = prob + (1 - valid_mask)
#
#             # get the prob of relevant label
#             label_onehot = F.one_hot(label, c)
#             label_onehot = label_onehot.transpose(1, 0)
#             prob = prob * label_onehot
#             prob = torch.sum(prob, dim=0)
#
#             threshold = self.thresh
#             if self.min_kept > 0:
#                 index = prob.argsort()
#                 threshold_index = index[min(len(index), self.min_kept) - 1]
#                 threshold_index = threshold_index[0]
#                 threshold_index = torch.tensor(threshold_index,dtype=torch.int64)
#                 if prob[threshold_index] > self.thresh:
#                     threshold = prob[threshold_index]
#                 kept_mask = torch.tensor((prob < threshold), dtype=torch.int64)  # .astype('int64')
#                 label = label * kept_mask
#                 valid_mask = valid_mask * kept_mask
#
#         # make the invalid region as ignore
#         label = label + (1 - valid_mask) * self.ignore_index
#
#         label = label.view((n, 1, h, w))
#         valid_mask = torch.tensor(valid_mask.view((n, 1, h, w)), dtype=torch.float32)  # .astype('float32')
#         loss = F.cross_entropy(logit, label, ignore_index=self.ignore_index)
#         loss = loss * valid_mask
#         avg_loss = torch.mean(loss) / (torch.mean(valid_mask) + self.EPS)
#
#         label.stop_gradient = True
#         valid_mask.stop_gradient = True
#         return avg_loss


class OhemCELoss(nn.Module):
    def __init__(self, thresh, *args, **kwargs, ):
        super(OhemCELoss, self).__init__()
        self.ignore_lb = 255
        self.n_min = None
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.criteria = nn.CrossEntropyLoss(ignore_index=self.ignore_lb, reduction='none')

    def forward(self, logits, labels):
        # labels = torch.tensor(labels, dtype=torch.uint8)
        # if torch.any(labels == 255):
        #     ignore_index = 255
        # else:
        #     ignore_index = -1
        # labels = torch.tensor(labels, dtype=torch.int64)
        # # N, C, H, W = logits.size()
        # self.ignore_lb = ignore_index
        self.n_min = labels[labels != 255].numel() // 16
        # self.n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels.long())
        # loss = self.criteria(logits, labels)
        loss = loss.view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


if __name__ == '__main__':
    pass
    #  criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    #  criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
