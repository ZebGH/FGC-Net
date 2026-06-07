import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

class DeformableStripConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(1, 9)):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (0, kernel_size[1] // 2) # 左右padding，保持宽度不变

        # 1. 偏移量预测层: 一个标准卷积，输出通道数为 2 * Kx * Ky
        # Kx, Ky 是卷积核尺寸。2代表(x, y)两个方向的偏移。
        # 这个层的权重会在训练中学习如何根据输入特征来预测最佳采样点位置。
        self.offset_conv = nn.Conv2d(c_in, 2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=3, padding=1, stride=1)

        # 2. 可变形卷积层本体
        self.dcn = DeformConv2d(c_in, c_out, kernel_size=kernel_size,
                                padding=padding, stride=1, bias=False)

    def forward(self, x):
        # 首先，从输入特征x中预测出偏移量
        offsets = self.offset_conv(x)

        # 然后，将输入特征x和预测的偏移量一同送入可变形卷积层
        # DCN会根据offsets去采样x，然后进行卷积
        return self.dcn(x, offsets)