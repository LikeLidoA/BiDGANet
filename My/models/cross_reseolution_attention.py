import math

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from models.blocks import ConvBNReLU


class ExternalAttention(nn.Module):

    def __init__(self, d_model=512, S=64):
        super().__init__()

        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        queries = x.view(b, c, n)  # 即bs，n，d_model
        queries = queries.permute(0, 2, 1)
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / (1e-9 + torch.sum(attn, dim=2, keepdim=True))  # bs,n,S
        attn = self.mv(attn)  # bs,n,d_model
        attn = attn.permute(0, 2, 1)
        x_attn = attn.view(b, c, h, w)
        x = x + x_attn
        x = F.relu(x)
        return x


class CrossResolutionAttention(nn.Module):

    def __init__(self, chann_high=128, chann_low=128):
        super().__init__()
        self.EAlayer = ExternalAttention(chann_high + chann_low)
        self.channel_high = chann_high
        self.channel_low = chann_low
        self.CBNReLU = ConvBNReLU(128, 128, 3, stride=2)
        self.init_weights()

    def init_weights(self):
        for m in self.CBNReLU():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x_h, x_l):
        x_h_in = self.CBNReLU(x_h)
        x1 = torch.cat((x_h_in, x_l), 1)
        x_l_in = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x_l)
        x2 = torch.cat((x_h, x_l_in), 1)
        x_l = self.EAlayer(x1)
        x_h = self.EAlayer(x2)
        x_high = x_h[:, :self.channel_high, :, :]
        x_low = x_l[:, self.channel_high:(self.channel_high + self.channel_low), :, :]
        return x_high, x_low


if __name__ == '__main__':
    input_h = torch.ones(2, 128, 128, 256)
    input_l = torch.randn(2, 256, 64, 128)
    channel_high = input_h.size(1)
    channel_low = input_l.size(1)
    # input_x = torch.cat((input_h, input_l), 1)
    Cross_Atten = CrossResolutionAttention(channel_high, channel_low)
    output_high, output_low = Cross_Atten(input_h, input_l)

    print(output_low.shape)
