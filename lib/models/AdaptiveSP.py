import torch
import torch.nn as nn

class AdaptiveSplit(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.nc = nc
        # 可学习的分割点参数（范围：0到nc）
        self.s1 = nn.Parameter(torch.tensor(0.0))  # 分割点1（a的结束位置）
        self.s2 = nn.Parameter(torch.tensor(0.0))  # 分割点2（b的结束位置）

    def forward(self, x):
        # 计算分割点（归一化到0到nc）
        s1 = torch.sigmoid(self.s1) * self.nc
        s2 = torch.sigmoid(self.s2) * self.nc

        # 确保 s1 < s2
        s1, s2 = torch.sort(torch.stack([s1, s2]))[0]
        s1, s2 = s1.item(), s2.item()

        # 计算通道数
        a = int(s1)
        b = int(s2 - s1)
        c = self.nc - a - b

        # 动态分割通道
        task1 = x[:, :a]
        task2 = x[:, a:a+b]
        task3 = x[:, a+b:]

        return task1, task2, task3