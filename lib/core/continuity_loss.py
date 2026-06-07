# lib/core/continuity_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuityLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, features, segmentation_logits):
        """
        Calculates a loss that encourages feature continuity for lane pixels.

        Args:
            features (Tensor): The feature map before the final classification layer.
                               Shape: [B, C, H, W]
            segmentation_logits (Tensor): The final output logits from the segmentation head.
                                          Shape: [B, Num_Classes, H, W]
        """
        # 1. 确定哪些像素被预测为车道线 (ID 1: 实线, ID 2: 虚线)
        # 我们只在这些像素上计算连续性损失
        _, pred_mask = torch.max(segmentation_logits, 1)  # Shape: [B, H, W]
        is_lane_mask = (pred_mask > 0).float().unsqueeze(1)  # Shape: [B, 1, H, W]

        # 2. 计算水平和垂直方向上相邻像素的特征差异
        # 使用L1损失来计算差异，它对异常值不那么敏感
        loss_h = F.l1_loss(features[:, :, :, :-1], features[:, :, :, 1:], reduction='none')
        loss_v = F.l1_loss(features[:, :, :-1, :], features[:, :, 1:, :], reduction='none')

        # 3. 只在被预测为车道线的像素位置上应用这个损失
        # 我们只关心车道线区域内部的连续性
        continuity_loss_h = (loss_h * is_lane_mask[:, :, :, :-1]).mean()
        continuity_loss_v = (loss_v * is_lane_mask[:, :, :-1, :]).mean()

        total_continuity_loss = continuity_loss_h + continuity_loss_v

        return total_continuity_loss * self.weight