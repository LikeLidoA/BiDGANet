import torch.nn as nn
import torch
from . import functional as Fu
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                input=score, size=(h, w), mode='bilinear')
        loss = self.criterion(score, target)

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, thres=0.7,
                 min_kept=25600, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()


class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.nll_loss = nn.NLLLoss(weight, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(self.alpha * (1 - F.softmax(inputs, dim=1)) ** self.gamma * F.log_softmax(inputs, dim=1),
                             targets)


class JaccardLoss(nn.Module):

    def __init__(self, eps=1., ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = F.softmax(y_pr, dim=1)
        return 1 - Fu.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(nn.Module):

    def __init__(self, eps=1., beta=1., ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = F.softmax(y_pr, dim=1)
        return 1 - Fu.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class Lovasz_Softmax(nn.Module):
    """
	Multi-class Lovasz-Softmax loss
	  probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
			  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
	  labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
	  classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
	  per_image: compute the loss per image instead of per batch
	  ignore: void class labels
	"""

    def __init__(self, ignore_label=255, classes='present', per_image=False):
        super(Lovasz_Softmax, self).__init__()
        self.ignore = ignore_label
        self.classes = classes
        self.per_image = per_image

    def forward(self, probas, labels):
        if self.per_image:
            loss = mean(Fu.lovasz_softmax_flat(*Fu.flatten_probas(prob.unsqueeze(0),
                                                                  lab.unsqueeze(0), self.ignore), classes=self.classes)
                        for prob, lab in zip(probas, labels))
        else:
            loss = Fu.lovasz_softmax_flat(*Fu.flatten_probas(probas, labels, self.ignore), classes=self.classes)
        return loss


class CategoricalCELoss(nn.Module):

    def __init__(self, class_weights=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights if class_weights is not None else 1
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = F.softmax(y_pr, dim=1)
        return Fu.categorical_crossentropy(
            y_pr, y_gt,
            class_weights=self.class_weights,
            ignore_channels=self.ignore_channels,
        )


class CategoricalFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2., activation="softmax", ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_channels = ignore_channels
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return Fu.categorical_focal_loss(
            y_pr, y_gt,
            alpha=self.alpha,
            gamma=self.gamma,
            ignore_channels=self.ignore_channels,
        )
