import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# '''来自Unet的保持分辨率RE卷积层，空洞卷积的一种应用'''
class REBnConv(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, dilation_rate=1):
        super(REBnConv, self).__init__()

        self.conv_s1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1 * dilation_rate,
                                 dilation=1 * dilation_rate)
        self.bn_s1 = nn.BatchNorm2d(out_channel)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


# '''内部用的上采样,src表示上采后，tar表示上采前一层'''
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

    return src


# '''深层的反卷积上采'''
class ConvTransposeBnReLUDeep(nn.Module):
    def __init__(self, in_channel=512, out_channel=256, kernelsize=4, stride=2, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvTransposeBnReLUDeep, self).__init__()
        self.TransConv = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=kernelsize, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv2d = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1,
                                padding=1, dilation=dilation,
                                groups=1, bias=bias)
        self.conv2d_squeeze = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1,
                                        padding=0, dilation=dilation,
                                        groups=1, bias=bias)
        self.Bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feature_map = self.TransConv(x)
        feature_map = self.conv2d(feature_map)
        feature_map = self.conv2d_squeeze(feature_map)
        feature_map = self.Bn(feature_map)
        feature_map = self.relu(feature_map)
        return feature_map


# '''基本的CBR层'''
class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernelsize=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size=kernelsize, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feature_map = self.conv(x)
        feature_map = self.bn(feature_map)
        feature_map = self.relu(feature_map)
        return feature_map


# """分割头"""
class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
            ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


# "BGA改"
class BGALayer(nn.Module):

    def __init__(self, in_chan=256, mid_chan=128):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=4, dilation=4, bias=True),
            nn.BatchNorm2d(mid_chan),
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                256, mid_chan, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                256, in_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(in_chan, mid_chan, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up2 = nn.Upsample(scale_factor=2)

        self.conv = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),  # not shown in paper
        )

    def forward(self, x_h, x_l):
        # dsize = x_d.size()[2:]
        left1 = self.left1(x_h)
        left2 = self.left2(x_h)
        right1 = self.right1(x_l)
        right1 = self.up(right1)
        right2 = self.right2(x_l)
        right2 = self.up(right2)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.tanh(right2)
        # right = self.up(right)
        out = self.conv(left + right)
        return out


class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernelsize=1, stride=1, padding=0, dilation=1)
        self.enc = nn.Conv2d(c_mid, (scale * k_up) ** 2, kernel_size=k_enc, stride=1, padding=k_enc // 2, dilation=1)
        self.bn = nn.BatchNorm2d((scale * k_up) ** 2)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.bn(W)
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X
# 旧分割头
# class SegmentHead(nn.Module):
#
#     def __init__(self, in_chan, mid_chan, n_classes):
#         super(SegmentHead, self).__init__()
#         self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
#         self.drop = nn.Dropout(0.1)
#         self.conv_out = nn.Conv2d(
#             mid_chan, n_classes, kernel_size=(1, 1), stride=1,
#             padding=0, bias=True)
#
#     def forward(self, x, size=None):
#         feat = self.conv(x)
#         feat = self.drop(feat)
#         feat = self.conv_out(feat)
#         if not size is None:
#             feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=True)
#         return feat


# 不启用的快速下采块
# class StemBlock(nn.Module):
#     def __init__(self):
#         super(StemBlock, self).__init__()
#         self.conv = ConvBNReLU(3, 16, 3, stride=2)
#         self.left_path = nn.Sequential(
#             ConvBNReLU(16, 8, 1, stride=1, padding=0),
#             ConvBNReLU(8, 16, 3, stride=2),
#         )
#         self.right_path = nn.MaxPool2d(
#             kernel_size=3, stride=2, padding=1, ceil_mode=False)
#         self.fuse = ConvBNReLU(32, 16, 3, stride=1)
#
#     def forward(self, x):
#         feature_map = self.conv(x)
#         feat_left = self.left_path(feature_map)
#         feat_right = self.right_path(feature_map)
#         feature_map = torch.cat([feat_left, feat_right], dim=1)
#         feature_map = self.fuse(feature_map)
#         return feature_map
