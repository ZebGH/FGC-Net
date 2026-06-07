# lib/core/dice_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=0, smooth=1e-6):
        """
        Multi-class Dice Loss.
        Ignores the background class by default.

        Args:
            ignore_index (int): Class index to ignore. Default is 0 for background.
            smooth (float): A small value to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Raw model output of shape [B, C, H, W].
            targets (Tensor): Ground truth of shape [B, H, W] with class IDs.
        """
        num_classes = logits.shape[1]

        # 1. Get probabilities with softmax
        probs = F.softmax(logits, dim=1)

        # 2. Convert targets to one-hot encoding
        # Shape: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        loss = 0.0
        # 3. Calculate Dice loss for each class (except ignored ones) and average
        for i in range(num_classes):
            if i == self.ignore_index:
                continue  # Skip background class

            pred_class = probs[:, i, :, :]
            target_class = one_hot_targets[:, i, :, :]

            intersection = torch.sum(pred_class * target_class)
            pred_sum = torch.sum(pred_class)
            target_sum = torch.sum(target_class)

            # Dice coefficient for this class
            dice_coeff = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

            # Add class-wise loss
            loss += (1. - dice_coeff)

        # Average the loss over the number of foreground classes
        num_foreground_classes = num_classes - 1 if self.ignore_index is not None else num_classes
        return loss / num_foreground_classes if num_foreground_classes > 0 else 0.0