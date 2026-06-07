import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


# ==========================================
# 前置依赖模块 (如果你的库里已有标准的 Conv，可以复用你的)
# ==========================================
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    # 标准卷积块
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    # 标准残差块 (用于刚性特征)
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_DCN(nn.Module):
    # DCN 残差块 (用于柔性形变特征)
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)

        # 替换为 DCN 卷积
        self.offset_conv = nn.Conv2d(c_, 2 * 3 * 3, kernel_size=3, padding=1, stride=1)
        self.dcn = DeformConv2d(c_, c2, kernel_size=3, padding=1, stride=1, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv1(x)
        offset = self.offset_conv(out)
        out = self.act(self.bn(self.dcn(out, offset)))
        return x + out if self.add else out


# ==========================================
# C2f 基础模块构建
# ==========================================
class C2f(nn.Module):
    # 纯刚性 C2f (YOLOv8 标配)
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_DCN(nn.Module):
    # 纯柔性 C2f_DCN
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_DCN(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ==========================================
# 🌟 本文终极杀器：双轨制解耦模块 🌟
# ==========================================
class HF_C2f_DCN(nn.Module):
    """
    High-Frequency Preserving C2f-DCN (高频保真可变形卷积块)
    通过通道正交解耦，同时保留刚性边缘与柔性语义。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        # 通道平分，保证参数量和计算量不爆炸
        c_rigid = c2 // 2
        c_deform = c2 - c_rigid

        # 路径1：刚性高频路径 (捕捉清晰的 Bounding Box 边缘)
        self.rigid_branch = C2f(c1, c_rigid, n, shortcut, g, e)

        # 路径2：柔性形变路径 (捕捉被遮挡和变形的车辆主体)
        self.deform_branch = C2f_DCN(c1, c_deform, n, shortcut, g, e)

    def forward(self, x):
        # 提取两种维度的特征
        feat_rigid = self.rigid_branch(x)
        feat_deform = self.deform_branch(x)

        # 在通道维度完美拼接，送入下一层
        return torch.cat((feat_rigid, feat_deform), dim=1)