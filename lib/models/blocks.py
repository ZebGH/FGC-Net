import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torch.nn.functional as F
import warnings






# class Conv(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=nn.SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Conv, self).__init__()
#         self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
#         self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#     def fuseforward(self, x):
#         return self.act(self.conv(x))

# class SPPCSPC(nn.Module):
#     # CSP https:///WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
#         super(SPPCSPC, self).__init__()
#         c_ = int(2 * c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(c_, c_, 3, 1)
#         self.cv4 = Conv(c_, c_, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
#         self.cv5 = Conv(4 * c_, c_, 1, 1)
#         self.cv6 = Conv(c_, c_, 3, 1)
#         self.cv7 = Conv(2 * c_, c2, 1, 1)
#
#     def forward(self, x):
#         x1 = self.cv4(self.cv3(self.cv1(x)))
#         y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
#         y2 = self.cv2(x)
#         return self.cv7(torch.cat((y1, y2), dim=1))

class Bconv(nn.Module):
    def __init__(self,ch_in,ch_out,k,s):
        super(Bconv, self).__init__()
        self.conv=nn.Conv2d(ch_in,ch_out,k,s,padding=k//2)
        self.bn =nn.BatchNorm2d(ch_out)
        self.act=nn.SiLU()
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))



class SppCSPC(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SppCSPC, self).__init__()
        # 分支一
        self.conv1 = nn.Sequential(
            Bconv(ch_in, ch_out, 1, 1),
            Bconv(ch_out, ch_out, 3, 1),
            Bconv(ch_out, ch_out, 1, 1)
        )
        # 分支二（SPP）
        self.mp1 = nn.MaxPool2d(5, 1, 5 // 2)  # 卷积核为5的池化
        self.mp2 = nn.MaxPool2d(9, 1, 9 // 2)  # 卷积核为9的池化
        self.mp3 = nn.MaxPool2d(13, 1, 13 // 2)  # 卷积核为13的池化
        # concat之后的卷积
        self.conv1_2 = nn.Sequential(
            Bconv(4 * ch_out, ch_out, 1, 1),
            Bconv(ch_out, ch_out, 3, 1)
        )
        # 分支三
        self.conv3 = Bconv(ch_in, ch_out, 1, 1)
        # 此模块最后一层卷积
        self.conv4 = Bconv(2 * ch_out, ch_out, 1, 1)
    def forward(self, x):
        # 分支一输出
        output1 = self.conv1(x)
        # 分支二池化层的各个输出
        mp_output1 = self.mp1(output1)
        mp_output2 = self.mp2(output1)
        mp_output3 = self.mp3(output1)
        # 合并以上并进行卷积
        result1 = self.conv1_2(torch.cat((output1, mp_output1, mp_output2, mp_output3), dim=1))
        # 分支三
        result2 = self.conv3(x)
        return self.conv4(torch.cat((result1, result2), dim=1))


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False, padding=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision,
                      relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False, padding=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2,
                      dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False, padding=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1,
                      groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4,
                      dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False, padding=1)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

# class ASPP(nn.Module):
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super(ASPP, self).__init__()
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
#         # self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
#         self.branch5_relu = nn.ReLU(inplace=True)
#
#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         [b, c, row, col] = x.size()
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)
#         conv3x3_2 = self.branch3(x)
#         conv3x3_3 = self.branch4(x)
#
#         global_feature = torch.mean(x, 2, True)
#         global_feature = torch.mean(global_feature, 3, True)
#         global_feature = self.branch5_conv(global_feature)
#         # global_feature = self.branch5_bn(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#         global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
#
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#         result = self.conv_cat(feature_cat)
#         return result


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 修改 Branch 5: 移除 BatchNorm
        self.branch5_pool = nn.AdaptiveAvgPool2d((1, 1)) # 使用 AdaptiveAvgPool2d 更标准
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        # self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom) # <-- 移除这一行
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        # 修改 Branch 5 的计算流程
        global_feature = self.branch5_pool(x)
        global_feature = self.branch5_conv(global_feature)
        # global_feature = self.branch5_bn(global_feature) # <-- 移除对BN的调用
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, size=(row, col), mode='bilinear', align_corners=True) # align_corners=True 通常与bilinear一起使用

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f_linear = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.focal_layers = nn.ModuleList()
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[1]

        # pre linear projection
        x = self.f_linear(x).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # context aggregation
        ctx_all = 0.0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        x_out = q * self.h(ctx_all)
        x_out = x_out.contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.stack(x).sum(dim=0)
        return y
# class Bottleneck(nn.Module):
#     """Standard bottleneck."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
#         expansion.
#         """
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         """'forward()' applies the YOLO FPN to input data."""
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#
# class C2f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""
#
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#
#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
#
#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))




def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """自动计算padding以保持特征图尺寸不变"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """标准卷积块 (Conv + BatchNorm + SiLU)"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """YOLOv8中的标准Bottleneck模块"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 如果shortcut为True且输入输出通道数相同，则执行残差连接
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    更快速的CSP Bottleneck，带有2个卷积层 (YOLOv8核心模块)
    参考自ultralytics/yolov8
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        # 初始卷积层，将输入通道c1转换为2*c，为后续分割做准备
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # 最后的卷积层，将拼接后的所有特征融合，并输出最终通道数c2
        # 输入通道数为 2*c (来自初始分割) + n*c (来自n个Bottleneck的输出)
        # 但在官方实现中，是初始分割的一半(c) + n个bottleneck的输出(n*c)
        # 我们遵循官方实现
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 官方代码是(2+n)*c，这里保持一致

        # 创建n个串联的Bottleneck模块
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        # 1. 初始卷积和分割
        # x: [B, c1, H, W] -> [B, 2*c, H, W]
        # y: list of 2 tensors, each [B, c, H, W]
        y = list(self.cv1(x).chunk(2, 1))

        # 2. 流经n个Bottleneck模块，并将每个模块的输出都收集起来
        # y[-1]始终是上一个模块的输出
        # y的长度会从2增长到2+n
        y.extend(m(y[-1]) for m in self.m)

        # 3. 将所有收集到的特征图在通道维度上进行拼接
        # y[0]是初始分割的第一个分支
        # y[1]是初始分割的第二个分支
        # y[2]...y[2+n-1]是n个Bottleneck的输出
        # 在官方代码中，y[1]实际上没有参与最终的拼接，而是作为bottleneck的第一个输入
        # 我们这里为了代码简洁和理解，直接用y.extend，其效果等价
        # 最终拼接的张量数量是 n+2 个
        # 这里需要注意，官方实现中，y.extend(m(y[-1]) for m in self.m)
        # 的逻辑是 y = [split1, split2], for m in ms: y.append(m(y[-1]))
        # 这意味着 y[-1] 在循环中是动态变化的。
        # 比如n=2:
        # y = [s1, s2]
        # y.append(m1(s2)) -> y = [s1, s2, m1_out]
        # y.append(m2(m1_out)) -> y = [s1, s2, m1_out, m2_out]
        # 最后拼接的是y中所有的元素

        # 4. 最终卷积融合
        return self.cv2(torch.cat(y, 1))

    # 示例：一个轻量级的大核卷积模块
class LargeKernelBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        # 使用7x7深度可分离卷积来获得大感受野，同时保持较低的计算量
        self.dwconv = nn.Conv2d(c_in, c_in, kernel_size=7, padding=3, groups=c_in)  # depthwise
        self.pwconv = nn.Conv2d(c_in, c_out, kernel_size=1)  # pointwise
        self.act = nn.GELU()  # 或 SiLU
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.act(x)
        x = self.pwconv(x)
        x = self.bn(x)
        return x

class Upsample_PixelShuffle(nn.Module):
    """
    使用PixelShuffle的可学习上采样模块
    """
    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()
        # 卷积层的输出通道需要是 目标通道数 * (放大倍数的平方)
        self.conv = Conv(c1, c2 * (scale_factor ** 2), k=1)
        # PixelShuffle层，负责重排像素
        self.shuffler = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        return self.shuffler(self.conv(x))
if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = ASPP(512, 512).cuda()
    out = model(x)
    print(out.shape)