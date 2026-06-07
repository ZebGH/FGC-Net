import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Computes the Sørensen–Dice loss.
    Operates directly on logits for numerical stability.
    Assumes binary segmentation (output channels = 1 or 2 after sigmoid/softmax).
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, *) tensor representing raw network outputs (before sigmoid/softmax).
                    For binary segmentation, expects (N, H, W) or (N, 1, H, W).
            targets: (N, *) tensor representing ground truth labels (0 or 1).
                     Should have the same shape as logits after potential channel squeezing.
        Returns:
            Dice loss.
        """
        num_classes = logits.shape[1] if len(logits.shape) > 3 else 1 # Check if channel dim exists

        if num_classes == 1:
             # Binary case: apply sigmoid and treat as single class foreground
            probs = torch.sigmoid(logits)
            targets = targets.float() # Ensure target is float
            # Flatten spatial dimensions N, H, W -> N, H*W
            probs = probs.view(probs.size(0), -1)
            targets = targets.view(targets.size(0), -1)
        else:
            # Potential multi-class case (though DA/LL seem binary based on old code)
            # If your output is (N, C, H, W) and targets are (N, H, W) integer labels
            # You might need one-hot encoding for targets and softmax for probs
            # Assuming binary based on original BCEseg usage:
            # If logits are (N, 2, H, W) for background/foreground
            probs = F.softmax(logits, dim=1)[:, 1] # Get foreground probability
            targets = targets.float() # Ensure target is float (might need one-hot if C > 2)
            probs = probs.view(probs.size(0), -1)
            targets = targets.view(targets.size(0), -1)


        intersection = torch.sum(probs * targets, dim=1)
        pred_sum = torch.sum(probs, dim=1)
        target_sum = torch.sum(targets, dim=1)

        # Dice Coefficient
        dice_coeff = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # Dice Loss
        dice_loss = 1. - dice_coeff

        return dice_loss.mean() # Return mean loss over the batch