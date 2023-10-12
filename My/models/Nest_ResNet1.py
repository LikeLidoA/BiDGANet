import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import REBnConv
from models.blocks import _upsample_like
from models.blocks import SegmentHead


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

    def __init__(self, number_of_class=19, in_ch=3, out_ch=1):
        super(ReHalf_U2NET, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)
        self.Head = SegmentHead(512, 1024, number_of_class)
        self.init_weights()
        # decoder
        # self.stage5d = RSU4F(1024, 256, 512)
        # self.stage4d = RSU4(1024, 128, 256)
        # self.stage3d = RSU5(512, 64, 128)
        # self.stage2d = RSU6(256, 32, 64)
        # self.stage1d = RSU7(128, 16, 64)
        #
        # self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        # self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        # self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        # self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        # self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        # self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        #
        # self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

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
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        # hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        # hx5dup = _upsample_like(hx5d, hx4)
        #
        # hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        # hx4dup = _upsample_like(hx4d, hx3)
        #
        # hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        # hx3dup = _upsample_like(hx3d, hx2)
        #
        # hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        # hx2dup = _upsample_like(hx2d, hx1)
        #
        # hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        # d1 = self.side1(hx1d)
        #
        # d2 = self.side2(hx2d)
        # d2 = _upsample_like(d2, d1)
        #
        # d3 = self.side3(hx3d)
        # d3 = _upsample_like(d3, d1)
        #
        # d4 = self.side4(hx4d)
        # d4 = _upsample_like(d4, d1)
        #
        # d5 = self.side5(hx5d)
        # d5 = _upsample_like(d5, d1)
        #
        # d6 = self.side6(hx6)
        # d6 = _upsample_like(d6, d1)
        #
        # d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        hx_fe = hx6up + hx5
        feature = self.Head(hx_fe, size)
        # m = F.softmax(feature,dim=1)
        # return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
        return feature# ,hx1,hx2,hx3,hx4,hx5,hx_fe

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
    x = torch.randn(2, 3, 128, 256).to(device)
    model = ReHalf_U2NET(number_of_class=19).cuda(device)
    net = model(x)[0]
    print(net)
    print(net.size())
    # for name, param in model.named_parameters():
    #     if len(param.size()) == 1:
    #         print(name)
