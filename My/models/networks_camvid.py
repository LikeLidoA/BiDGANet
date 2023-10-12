import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import ConvBNReLU
# from models.Nest_ResNet1 import ReHalf_U2NET
from models.Nest_ResNet3 import ReHalf_U2NET
from models.blocks import SegmentHead, CARAFE
from models.cross_reseolution_attention2 import CrossResolutionAttention


# from models.blocks import ConvTransposeBnReLUDeep


# from models.blocks import BGALayer


class HighResolutionBranch(nn.Module):

    def __init__(self):
        super(HighResolutionBranch, self).__init__()
        self.Stage1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.Stage2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.Stage3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        # self.Head = SegmentHead(128, 256, 19)
        self.init_weights()

    def forward(self, x):
        size = x.size()[2:]
        feature_map = self.Stage1(x)
        feature_map = self.Stage2(feature_map)
        feature_map = self.Stage3(feature_map)
        # feature_map = self.Head(feature_map, size)
        return feature_map

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class LowResolutionBranch(nn.Module):
    def __init__(self):
        super(LowResolutionBranch, self).__init__()
        self.net = ReHalf_U2NET()
        # pretrained_net = torch.load("")
        # self.net.load_state_dict(pretrained_net)

    def forward(self, x):
        feature_map = self.net(x)
        return feature_map


class Mymodel(nn.Module):

    def __init__(self):
        super(Mymodel, self).__init__()

        self.highresolution = HighResolutionBranch()
        self.lowresolution = LowResolutionBranch()
        self.Cross_Atten = CrossResolutionAttention(128, 256)
        # self.TranConv = ConvTransposeBnReLUDeep(512, 128)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2times1 = CARAFE(256)
        self.up2times2 = CARAFE(384)
        # self.bga = BGALayer(128, 256)
        self.Head = SegmentHead(384, 512, 19, up_factor=4, aux=False)

        # for m in self.bga():
        #     if isinstance(m, nn.Conv2d):
        #         init.kaiming_normal_(m.weight, mode='fan_out')
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init.constant_(m.weight, 1)
        #         init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         init.normal_(m.weight, std=0.001)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)

    def forward(self, x):
        # 主干模型部分
        feature_h = self.highresolution(x)  # 高分辨率分支输出的特征图
        feature_l = self.lowresolution(x)  # 低分辨率分支输出的特诊图

        # channel_deep = feature_l.size(1)  # 深层下采最终输出层的通道数
        # channel_high = feature_h.size(1)  # 高分辨率分支的通道数
        # channel_low = feature_for_atten.size(1)  # 低分辨率分支用于进行跨分辨注意力计算的层的通道数
        # 中间操作
        # feature_l = self.TranConv(feature_l)  # 把最深层输出转置上采到和跨分辨注意力输出一样的尺寸和通道数
        # feature_l = self.up(feature_l)
        # 准备进入跨分辨率注意力模块
        # 注意力部分
        att_output_high, att_output_low = self.Cross_Atten(feature_h, feature_l)  # 分别输出给高分辨分支和低分辨分支的注意力图
        # 融合部分
        feature_h = feature_h + att_output_high  # 高分辨分支加上注意力

        # 低分辨分支的逐元素加（融合）
        feature_l = feature_l + att_output_low  # 低分辨分支加上注意力
        feature_l = self.up2times1(feature_l)
        # feature_end = self.bga(feature_h, feature_l)
        feature_end = torch.cat((feature_h, feature_l), 1)
        feature_end = self.up2times2(feature_end)
        output = self.Head(feature_end)  # 分割头
        return output


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 1024).cuda()
    model = Mymodel().cuda()
    outs = model(x)

    for out in outs:
        print(out.size())
