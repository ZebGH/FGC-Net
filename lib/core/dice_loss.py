# lib/core/dice_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=0, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        loss = 0.0
        num_foreground_classes = 0
        for i in range(num_classes):
            if i == self.ignore_index:
                continue

            pred_class = probs[:, i, :, :]
            target_class = one_hot_targets[:, i, :, :]

            intersection = torch.sum(pred_class * target_class)
            pred_sum = torch.sum(pred_class)
            target_sum = torch.sum(target_class)

            dice_coeff = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            loss += (1. - dice_coeff)
            num_foreground_classes += 1

        return loss / num_foreground_classes if num_foreground_classes > 0 else 0.0