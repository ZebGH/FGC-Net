import torch
import torch.nn as nn
import torchvision.ops as ops


class Conv(nn.Module):
    # 你的标准卷积模块 (假设已存在)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeformableConv2d(nn.Module):
    """标准的形变卷积 DCNv2 封装"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        # 学习偏移量 (x, y) 和 掩码 (mask)
        # 偏移量通道数: 2 * 3 * 3 = 18, 掩码通道数: 1 * 3 * 3 = 9. 共 27.
        self.offset_conv = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, kernel_size=3, padding=1)
        # 初始化偏移量为0
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, bias=bias)

    def forward(self, x):
        offset_and_mask = self.offset_conv(x)
        o1, o2, mask = torch.chunk(offset_and_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight,
                              bias=self.regular_conv.bias, padding=self.padding,
                              mask=mask, stride=self.stride)
        return x


class DeformableBottleneck(nn.Module):
    """将普通的 3x3 卷积替换为 DCN 的瓶颈层"""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1 降维压缩，减小DCN计算量
        # 💡 这里使用可变形卷积代替普通卷积
        self.cv2 = DeformableConv2d(c_, c2, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.bn(self.cv2(self.cv1(x)))
        out = self.act(out)
        return x + out if self.add else out


class C3_DCN(nn.Module):
    """Deformable CSP Bottleneck 模块 (高性价比)"""

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU or SiLU
        # 嵌套我们刚才写的 DeformableBottleneck
        self.m = nn.Sequential(*(DeformableBottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))