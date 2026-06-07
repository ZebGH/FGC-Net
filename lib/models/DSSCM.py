import torch
import torch.nn as nn


class DetSegSpatialConsistencyModule(nn.Module):
    """
    自适应空间一致性与边界向导模块 (Adaptive Spatial Consistency & Boundary Guidance)
    """

    def __init__(self, det_channels, da_channels, ll_channels):
        super().__init__()
        # 1. 检测遮挡提取头 (提取车辆位置)
        self.presence_head = nn.Sequential(
            nn.Conv2d(det_channels, det_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(det_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(det_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 2. 💡 车道线边界向导头
        self.ll_guide_head = nn.Sequential(
            nn.Conv2d(ll_channels, ll_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ll_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ll_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 可学习参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 车辆对路面的遮挡程度
        self.beta = nn.Parameter(torch.tensor(0.5))  # 车辆对车道线的遮挡程度
        self.gamma = nn.Parameter(torch.tensor(0.0))  # 💡 LL对DA的边界向导权重 (0.0起步，让数据去驱动)

        self.refine_da = nn.Conv2d(da_channels, da_channels, 1)
        self.refine_ll = nn.Conv2d(ll_channels, ll_channels, 1)

    def forward(self, x):
        d_prime, s_da_prime, s_ll_prime = x[0], x[1], x[2]

        det_presence_map = self.presence_head(d_prime)

        # 1. 车辆遮挡：保持软抑制
        s_da_modulated = s_da_prime * (1.0 - self.alpha * det_presence_map)
        s_ll_modulated = s_ll_prime * (1.0 - self.beta * det_presence_map)

        # 2. 💡 边界向导：用车道线去辅助勾勒可行驶区域
        ll_guide_map = self.ll_guide_head(s_ll_modulated)
        s_da_modulated = s_da_modulated + self.gamma * (s_da_modulated * ll_guide_map)

        # 细化输出
        s_da_modulated = self.refine_da(s_da_modulated)
        s_ll_modulated = self.refine_ll(s_ll_modulated)

        return s_da_modulated, s_ll_modulated