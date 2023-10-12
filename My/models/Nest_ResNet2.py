import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import REBnConv
from models.blocks import _upsample_like
from models.blocks import SegmentHead
from models.blocks import ConvBNReLU


class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.REBnConvin = REBnConv(in_ch, out_ch, dilation_rate=1)

        self.REBnConv1 = REBnConv(out_ch, mid_ch, dilation_rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv2 = REBnConv(mid_ch, mid_ch, dilation_rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv3 = REBnConv(mid_ch, mid_ch, dilation_rate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv4 = REBnConv(mid_ch, mid_ch, dilation_rate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv5 = REBnConv(mid_ch, mid_ch, dilation_rate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv6 = REBnConv(mid_ch, mid_ch, dilation_rate=1)

        self.REBnConv7 = REBnConv(mid_ch, mid_ch, dilation_rate=2)

        self.REBnConv6d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv5d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv4d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv3d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv2d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv1d = REBnConv(mid_ch * 2, out_ch, dilation_rate=1)

    def forward(self, x):
        hx = x
        hx_input = self.REBnConvin(hx)

        hx1 = self.REBnConv1(hx_input)
        hx = self.pool1(hx1)

        hx2 = self.REBnConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.REBnConv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.REBnConv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.REBnConv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.REBnConv6(hx)

        hx7 = self.REBnConv7(hx6)

        hx6d = self.REBnConv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.REBnConv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.REBnConv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.REBnConv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.REBnConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.REBnConv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hx_input


class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.REBnConvin = REBnConv(in_ch, out_ch, dilation_rate=1)

        self.REBnConv1 = REBnConv(out_ch, mid_ch, dilation_rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv2 = REBnConv(mid_ch, mid_ch, dilation_rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv3 = REBnConv(mid_ch, mid_ch, dilation_rate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv4 = REBnConv(mid_ch, mid_ch, dilation_rate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv5 = REBnConv(mid_ch, mid_ch, dilation_rate=1)

        self.REBnConv6 = REBnConv(mid_ch, mid_ch, dilation_rate=2)

        self.REBnConv5d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv4d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv3d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv2d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv1d = REBnConv(mid_ch * 2, out_ch, dilation_rate=1)

    def forward(self, x):
        hx = x

        hx_input = self.REBnConvin(hx)

        hx1 = self.REBnConv1(hx_input)
        hx = self.pool1(hx1)

        hx2 = self.REBnConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.REBnConv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.REBnConv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.REBnConv5(hx)

        hx6 = self.REBnConv6(hx5)

        hx5d = self.REBnConv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.REBnConv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.REBnConv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.REBnConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.REBnConv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hx_input


class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.REBnConvin = REBnConv(in_ch, out_ch, dilation_rate=1)

        self.REBnConv1 = REBnConv(out_ch, mid_ch, dilation_rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv2 = REBnConv(mid_ch, mid_ch, dilation_rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv3 = REBnConv(mid_ch, mid_ch, dilation_rate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv4 = REBnConv(mid_ch, mid_ch, dilation_rate=1)

        self.REBnConv5 = REBnConv(mid_ch, mid_ch, dilation_rate=2)

        self.REBnConv4d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv3d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv2d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv1d = REBnConv(mid_ch * 2, out_ch, dilation_rate=1)

    def forward(self, x):
        hx = x

        hx_input = self.REBnConvin(hx)

        hx1 = self.REBnConv1(hx_input)
        hx = self.pool1(hx1)

        hx2 = self.REBnConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.REBnConv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.REBnConv4(hx)

        hx5 = self.REBnConv5(hx4)

        hx4d = self.REBnConv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.REBnConv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.REBnConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.REBnConv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hx_input


class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.REBnConvin = REBnConv(in_ch, out_ch, dilation_rate=1)

        self.REBnConv1 = REBnConv(out_ch, mid_ch, dilation_rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv2 = REBnConv(mid_ch, mid_ch, dilation_rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv3 = REBnConv(mid_ch, mid_ch, dilation_rate=1)

        self.REBnConv4 = REBnConv(mid_ch, mid_ch, dilation_rate=2)

        self.REBnConv3d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv2d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=1)
        self.REBnConv1d = REBnConv(mid_ch * 2, out_ch, dilation_rate=1)

    def forward(self, x):
        hx = x

        hx_input = self.REBnConvin(hx)

        hx1 = self.REBnConv1(hx_input)
        hx = self.pool1(hx1)

        hx2 = self.REBnConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.REBnConv3(hx)

        hx4 = self.REBnConv4(hx3)

        hx3d = self.REBnConv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.REBnConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.REBnConv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hx_input


class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.REBnConvin = REBnConv(in_ch, out_ch, dilation_rate=1)

        self.REBnConv1 = REBnConv(out_ch, mid_ch, dilation_rate=1)
        self.REBnConv2 = REBnConv(mid_ch, mid_ch, dilation_rate=2)
        self.REBnConv3 = REBnConv(mid_ch, mid_ch, dilation_rate=4)

        self.REBnConv4 = REBnConv(mid_ch, mid_ch, dilation_rate=8)

        self.REBnConv3d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=4)
        self.REBnConv2d = REBnConv(mid_ch * 2, mid_ch, dilation_rate=2)
        self.REBnConv1d = REBnConv(mid_ch * 2, out_ch, dilation_rate=1)

    def forward(self, x):
        hx = x

        hx_input = self.REBnConvin(hx)

        hx1 = self.REBnConv1(hx_input)
        hx2 = self.REBnConv2(hx1)
        hx3 = self.REBnConv3(hx2)

        hx4 = self.REBnConv4(hx3)

        hx3d = self.REBnConv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.REBnConv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.REBnConv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hx_input


class ReHalf_U2NET(nn.Module):

    def __init__(self, number_of_class=20, in_ch=3, out_ch=1):
        super(ReHalf_U2NET, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 32)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(32, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(128, 64, 256)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(256, 128, 256)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(256, 128, 256)
        # self.Head = SegmentHead(512, 512, number_of_class)

        # self.bn = nn.BatchNorm2d(512)
        # self.conv_gap = ConvBNReLU(512, 512, (1, 1), stride=1, padding=0)
        # self.conv_last = ConvBNReLU(512, 512, (3,3), stride=1)
        # self.poolGlobal = nn.AdaptiveAvgPool2d(1)

        self.init_weights()

    def forward(self, x):
        hx = x
        size = x.size()[2:]
        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)

        # hx_context = hx6
        #
        # hx_fe = self.poolGlobal(hx_context)
        # hx_fe = self.bn(hx_fe)
        # hx_fe = self.conv_gap(hx_fe)
        # hx_fe = hx_fe + hx_context
        # hx_fe = self.conv_last(hx_fe)
        hx_fe = _upsample_like(hx6, hx5)
        # hx_fe = _upsample_like(hx_fe, hx5)
        hx_fe = hx_fe + hx5
        # feature = self.Head(hx_fe, size)
        # m = F.softmax(feature,dim=1)
        # return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
        return hx_fe  # ,hx1,hx2,hx3,hx4,hx5,hx_fe

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.BatchNorm2d):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(3, 3, 128, 256).to(device)
    model = ReHalf_U2NET(number_of_class=19).cuda(device)
    output1 = model(x)
    print(output1)
    # net = model(x)[0]
    # print(net)
    # print(net.size())
    # # for name, param in model.named_parameters():
    #     if len(param.size()) == 1:
    #         print(name)
