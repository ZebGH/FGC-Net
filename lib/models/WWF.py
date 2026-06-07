import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFeatureFusion(nn.Module):
    """
    可学习权重的特征融合模块，用于BiFPN。
    对输入的多路特征图进行加权求和。
    """

    def __init__(self, num_inputs, epsilon=1e-4):
        super(WeightedFeatureFusion, self).__init__()
        self.num_inputs = num_inputs
        self.epsilon = epsilon

        # 为每个输入特征图定义一个可学习的权重
        # nn.Parameter使其能被PyTorch的优化器自动识别和更新
        self.weights = nn.Parameter(torch.ones(num_inputs, requires_grad=True))

    def forward(self, inputs):
        """
        Args:
            inputs (list of torch.Tensor): 一个包含多个特征图张量的列表

        Returns:
            torch.Tensor: 加权融合后的特征图
        """
        assert len(inputs) == self.num_inputs, "输入的特征图数量与模块初始化时不同"

        # 1. 权重归一化：使用ReLU确保权重为非负，然后除以总和进行归一化
        #    这是EfficientDet论文中提出的“Fast normalized fusion”
        weights = F.relu(self.weights)
        weights = weights / (torch.sum(weights, dim=0) + self.epsilon)

        # 2. 加权求和
        output = torch.zeros_like(inputs[0])
        for i in range(self.num_inputs):
            output += weights[i] * inputs[i]

        return output


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# 为了代码完整性，这里也需要一个BiFPN整体模块的定义
# 我们可以将一个BiFPN层（包含一次自顶向下和一次自底向上）封装起来
class BiFPN_Layer(nn.Module):
    def __init__(self, c_in_list, c_out):
        super().__init__()
        c_p2, c_p3, c_p4 = c_in_list

        # --- 1x1 Convs to unify channels ---
        self.p2_in_conv = Conv(c_p2, c_out, 1, 1)
        self.p3_in_conv = Conv(c_p3, c_out, 1, 1)
        self.p4_in_conv = Conv(c_p4, c_out, 1, 1)

        # --- Top-Down Path ---
        self.fusion_p3_td = WeightedFeatureFusion(num_inputs=2)  # P3_in + Upsampled(P4_td)
        self.conv_p3_td = Conv(c_out, c_out, 3, 1)

        self.fusion_p2_td = WeightedFeatureFusion(num_inputs=2)  # P2_in + Upsampled(P3_td)
        self.conv_p2_out = Conv(c_out, c_out, 3, 1)

        # --- Bottom-Up Path ---
        self.downsample_p2_out = Conv(c_out, c_out, 3, 2)
        self.fusion_p3_out = WeightedFeatureFusion(num_inputs=3)  # P3_in + P3_td + Downsampled(P2_out)
        self.conv_p3_out = Conv(c_out, c_out, 3, 1)

        self.downsample_p3_out = Conv(c_out, c_out, 3, 2)
        self.fusion_p4_out = WeightedFeatureFusion(num_inputs=2)  # P4_in + Downsampled(P3_out)
        self.conv_p4_out = Conv(c_out, c_out, 3, 1)

    def forward(self, inputs):
        p2_in, p3_in, p4_in = inputs

        # Unify channels
        p2_in = self.p2_in_conv(p2_in)
        p3_in = self.p3_in_conv(p3_in)
        p4_in = self.p4_in_conv(p4_in)

        # Top-Down
        p3_td = self.conv_p3_td(self.fusion_p3_td([p3_in, F.interpolate(p4_in, scale_factor=2)]))
        p2_out = self.conv_p2_out(self.fusion_p2_td([p2_in, F.interpolate(p3_td, scale_factor=2)]))

        # Bottom-Up
        p3_out = self.conv_p3_out(self.fusion_p3_out([p3_in, p3_td, self.downsample_p2_out(p2_out)]))
        p4_out = self.conv_p4_out(self.fusion_p4_out([p4_in, self.downsample_p3_out(p3_out)]))

        return p2_out, p3_out, p4_out


if __name__ == "__main__":
    # C2(B, 128, 80, 80), C3(B, 320, 40, 40), C4(B, 512, 20, 20)
    x1 = torch.rand(4, 128, 80, 80)
    x2 = torch.rand(4, 320, 40, 40)
    x3 = torch.rand(4, 512, 20, 20)

    tensors = [x1, x2, x3]
    model = BiFPN_Layer([128, 320, 512], 256)

    y = model(tensors)