import torch
import torch.nn as nn
import torch.nn.functional as F
from .general import bbox_iou
from .postprocess import build_targets
from lib.core.DiceLoss import DiceLoss
# 假设这些辅助函数/类已在别处定义或导入:
# from .general import bbox_iou
# from .postprocess import build_targets
# from .DiceLoss import DiceLoss # 确保 DiceLoss 类已定义
# from .FocalLoss import FocalLoss # 确保 FocalLoss 类已定义
# from .SmoothBCE import smooth_BCE # 确保 smooth_BCE 函数已定义

# Placeholder implementations for missing helpers (你需要用你自己的实际实现替换)
# ==================================================
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6): super().__init__(); self.smooth = smooth
#     def forward(self, logits, targets):
#         num_classes = logits.shape[1] if len(logits.shape) == 4 else 1
#         if num_classes > 1: probs = F.softmax(logits, dim=1)[:, 1] # Foreground prob for C=2
#         else: probs = torch.sigmoid(logits)
#         targets = targets.float()
#         probs = probs.view(probs.size(0), -1)
#         targets = targets.view(targets.size(0), -1)
#         intersection = torch.sum(probs * targets, dim=1)
#         pred_sum = torch.sum(probs, dim=1)
#         target_sum = torch.sum(targets, dim=1)
#         dice_coeff = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
#         return (1. - dice_coeff).mean()

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def smooth_BCE(eps=0.1): return 1.0 - 0.5 * eps, 0.5 * eps

# def bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
#      # Placeholder: Replace with your actual bbox_iou implementation supporting CIoU
#      # This is a simplified version returning dummy IoU for structure completeness
#     if box1.shape[1] != 4 or box2.shape[1] != 4: return torch.zeros(box1.shape[0], box2.shape[0], device=box1.device) # Handle empty cases
#     return torch.rand(box1.shape[0], box2.shape[0], device=box1.device) * 0.5 + 0.2 # Dummy IoU

# def build_targets(cfg, p, targets, model):
#      # Placeholder: Replace with your actual build_targets implementation
#      # Returns dummy targets for structure completeness
#      device = targets.device
#      nt = targets.shape[0]
#      tcls, tbox, indices, anch = [], [], [], []
#      gain = torch.ones(7, device=device)
#      ai = torch.arange(3, device=device).float().view(3, 1).repeat(1, nt) # anchor indices
#      targets = torch.cat((targets.repeat(3, 1, 1), ai[:, :, None]), 2) # append anchor indices
#
#      g = 0.5 # bias
#      off = torch.tensor([[0, 0],[1, 0],[0, 1],[-1, 0],[0, -1]], device=device).float() * g # offsets
#
#      for i in range(model.nl): # model.nl is number of detection layers
#          anchors = model.anchors[i] # Get anchors for this layer
#          gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]] # xyxy gain
#          t = targets * gain
#          if nt:
#              r = t[:, :, 4:6] / anchors[:, None] # wh ratio
#              j = torch.max(r, 1. / r).max(2)[0] < cfg.TRAIN.ANCHOR_THRESHOLD # compare
#              t = t[j] # filter
#
#              gxy = t[:, 2:4] # grid xy
#              gxi = gain[[2, 3]] - gxy # inverse
#              j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#              l, m = ((gxi % 1. < g) & (gxi > 1.)).T
#              j = torch.stack((torch.ones_like(j), j, k, l, m))
#              t = t.repeat((5, 1, 1))[j]
#              offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#          else:
#              t = targets[0]
#              offsets = 0
#
#          b, c = t[:, :2].long().T # image, class
#          gxy = t[:, 2:4] # grid xy
#          gwh = t[:, 4:6] # grid wh
#          gij = (gxy - offsets).long()
#          gi, gj = gij.T # grid xy indices
#
#          a = t[:, 6].long() # anchor indices
#          indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1))) # image, anchor, grid indices
#          tbox.append(torch.cat((gxy - gij, gwh), 1)) # box
#          anch.append(anchors[a]) # anchors
#          tcls.append(c) # class
#      return tcls, tbox, indices, anch

# ==================================================


class MultiHeadLossUW(nn.Module):
    """
    Multi-Head Loss with Uncertainty Weighting and combined Segmentation Loss (CE + Dice).
    (整合了检测损失计算循环的完整版本)
    """
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # --- Instantiate Base Loss Functions ---
        # Detection Losses
        cls_pos_weight = torch.tensor([cfg.LOSS.CLS_POS_WEIGHT], device=device)
        obj_pos_weight = torch.tensor([cfg.LOSS.OBJ_POS_WEIGHT], device=device)
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=obj_pos_weight)
        if cfg.LOSS.FL_GAMMA > 0:
            self.BCEcls = FocalLoss(self.BCEcls, cfg.LOSS.FL_GAMMA)
            self.BCEobj = FocalLoss(self.BCEobj, cfg.LOSS.FL_GAMMA)

        # Segmentation Losses
        seg_class_weights = None # Define class weights for CE if needed
        self.CELoss = nn.CrossEntropyLoss(weight=seg_class_weights, ignore_index=cfg.DATASET.get('IGNORE_LABEL', -100)) # Use get for safety
        self.DiceLoss = DiceLoss()
        self.seg_loss_alpha = cfg.LOSS.get('SEG_LOSS_ALPHA', 0.5) # Use get for safety

        # --- Uncertainty Weighting Parameters ---
        self.log_vars = nn.Parameter(torch.zeros(5, device=device)) # cls, obj, box, da_seg, ll_seg

    def _process_seg_target(self, target_original):
        """Helper to ensure target is [N, H, W] Long type."""
        if len(target_original.shape) == 4 and target_original.shape[1] != 1:
            target_processed = torch.argmax(target_original, dim=1)
        elif len(target_original.shape) == 4 and target_original.shape[1] == 1:
            target_processed = target_original.squeeze(1).long()
        elif len(target_original.shape) == 3:
            target_processed = target_original.long()
        else:
            raise ValueError(f"Unsupported Segmentation Target shape: {target_original.shape}")
        return target_processed

    def forward(self, predictions, targets, shapes, model):
        """
        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3], da_seg_logits, ll_seg_logits]
            targets: gts [det_targets, da_seg_targets, ll_seg_targets]
            shapes: shapes info (used potentially in build_targets)
            model: model instance (used for anchors, nc, gr, nl)

        Returns:
            total_loss: sum of all the weighted losses
            head_losses_unweighted: tuple containing unweighted losses for logging
        """
        # Initialize Detection Losses
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)

        # Build Detection Targets
        # Ensure model has 'nl' (number of detection layers) and 'na' (number of anchors) attributes if needed by build_targets


        # # Ensure model has 'anchors' attribute if needed by build_targets
        # # Ensure model has 'gr' (IoU ratio) attribute if needed
        # if not hasattr(model, 'nl'): model.nl = len(predictions[0]) # Infer if not present
        # if not hasattr(model, 'na'): model.na = predictions[0][0].shape[1] if len(predictions[0]) > 0 else 3 # Infer if not present
        # if not hasattr(model, 'gr'): model.gr = 1.0 # Default IoU ratio if not present
        # # You might need to fetch anchors from cfg or attach them to the model instance correctly
        # if not hasattr(model, 'anchors'):
        #      print("Warning: model.anchors not found for build_targets. Using dummy anchors.")
        #      # Example: Assuming 3 layers, 3 anchors per layer, stride [8, 16, 32]
        #      dummy_anchors = [
        #          torch.tensor([[10,13], [16,30], [33,23]], device=self.device) / 8,
        #          torch.tensor([[30,61], [62,45], [59,119]], device=self.device) / 16,
        #          torch.tensor([[116,90], [156,198], [373,326]], device=self.device) / 32
        #      ]
        #      model.anchors = dummy_anchors[:model.nl]

        if not hasattr(model, 'nl'): model.nl = len(predictions[0])  # Infer if not present
        # if not hasattr(model, 'na'): model.na = predictions[0][0].shape[1] if len(predictions[0]) > 0 else 3 # build_targets 可能不需要这个
        if not hasattr(model, 'gr'): model.gr = 1.0  # Default IoU ratio if not present

        tcls, tbox, indices, anchors = build_targets(self.cfg, predictions[0], targets[0], model)

        # Class label smoothing
        cp, cn = smooth_BCE(eps=self.cfg.LOSS.get('LABEL_SMOOTHING', 0.0)) # Use get for safety

        # --- Calculate Detection Losses (Loop over prediction layers) ---
        nt = 0  # number of targets found
        no = len(predictions[0])  # number of detection output layers
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # Balance weights P3-5 or P3-6

        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=self.device)  # target obj

            n = b.shape[0]  # number of targets in this layer
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression Loss (CIoU)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox_pred = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox_pred.T, tbox[i], x1y1x2y2=False, CIoU=True) # Ensure your bbox_iou calculates CIoU
                lbox += (1.0 - iou).mean() # Accumulate mean loss for this layer

                # Objectness Loss Target
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype) # IoU aware

                # Classification Loss
                if model.nc > 1: # Only compute cls loss if number of classes > 1
                    t = torch.full_like(ps[:, 5:], cn, device=self.device)
                    t[range(n), tcls[i]] = cp
                    lcls += self.BCEcls(ps[:, 5:], t) # Accumulate cls loss for this layer

            # Objectness Loss (applied to the entire layer's objectness predictions)
            lobj += self.BCEobj(pi[..., 4], tobj) * balance[i] # Accumulate obj loss for this layer


        # Apply detection loss gains (after accumulation, before UW)
        s = 3 / no # Scale loss by number of detection layers
        lcls *= self.cfg.LOSS.CLS_GAIN * s
        lobj *= self.cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.)
        lbox *= self.cfg.LOSS.BOX_GAIN * s

        # --- Calculate Segmentation Losses (as before) ---
        da_seg_logits = predictions[1]
        da_seg_targets_original = targets[1]
        da_seg_targets_int = self._process_seg_target(da_seg_targets_original)
        da_seg_targets_float = da_seg_targets_int.float()
        lseg_da_ce = self.CELoss(da_seg_logits, da_seg_targets_int)
        lseg_da_dice = self.DiceLoss(da_seg_logits, da_seg_targets_float)
        lseg_da = self.seg_loss_alpha * lseg_da_ce + (1.0 - self.seg_loss_alpha) * lseg_da_dice
        lseg_da *= self.cfg.LOSS.DA_SEG_GAIN

        ll_seg_logits = predictions[2]
        ll_seg_targets_original = targets[2]
        ll_seg_targets_int = self._process_seg_target(ll_seg_targets_original)
        ll_seg_targets_float = ll_seg_targets_int.float()
        lseg_ll_ce = self.CELoss(ll_seg_logits, ll_seg_targets_int)
        lseg_ll_dice = self.DiceLoss(ll_seg_logits, ll_seg_targets_float)
        lseg_ll = self.seg_loss_alpha * lseg_ll_ce + (1.0 - self.seg_loss_alpha) * lseg_ll_dice
        lseg_ll *= self.cfg.LOSS.LL_SEG_GAIN

        # --- Apply Uncertainty Weighting ---
        loss_cls_w = 0.5 * torch.exp(-self.log_vars[0]) * lcls + 0.5 * self.log_vars[0]
        loss_obj_w = 0.5 * torch.exp(-self.log_vars[1]) * lobj + 0.5 * self.log_vars[1]
        loss_box_w = 0.5 * torch.exp(-self.log_vars[2]) * lbox + 0.5 * self.log_vars[2]
        loss_da_seg_w = 0.5 * torch.exp(-self.log_vars[3]) * lseg_da + 0.5 * self.log_vars[3]
        loss_ll_seg_w = 0.5 * torch.exp(-self.log_vars[4]) * lseg_ll + 0.5 * self.log_vars[4]

        # --- Handle Conditional Training ---
        # (Zero out weighted losses based on cfg flags)
        if self.cfg.TRAIN.get('DET_ONLY', False) or self.cfg.TRAIN.get('ENC_DET_ONLY', False):
             loss_da_seg_w = torch.zeros_like(loss_da_seg_w) # Use zeros_like for safety
             loss_ll_seg_w = torch.zeros_like(loss_ll_seg_w)
             lseg_da = torch.zeros_like(lseg_da)
             lseg_ll = torch.zeros_like(lseg_ll)
        # ... (similar logic for SEG_ONLY, LANE_ONLY, DRIVABLE_ONLY, use zeros_like) ...
        if self.cfg.TRAIN.get('SEG_ONLY', False) or self.cfg.TRAIN.get('ENC_SEG_ONLY', False):
             loss_cls_w = torch.zeros_like(loss_cls_w)
             loss_obj_w = torch.zeros_like(loss_obj_w)
             loss_box_w = torch.zeros_like(loss_box_w)
             lcls = torch.zeros_like(lcls)
             lobj = torch.zeros_like(lobj)
             lbox = torch.zeros_like(lbox)
        # ... add others similarly ...


        # --- Calculate Total Loss ---
        total_loss = loss_cls_w + loss_obj_w + loss_box_w + loss_da_seg_w + loss_ll_seg_w

        # --- Prepare Losses for Logging ---
        # Return unweighted, gain-applied losses (before UW was applied)
        head_losses_unweighted = (lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), total_loss.item())

        return total_loss, head_losses_unweighted

def get_loss(cfg, device):
    """
    Get MultiHeadLoss with Uncertainty Weighting.

    Inputs:
    - cfg: configuration object
    - device: cpu or gpu device

    Returns:
    - loss: (MultiHeadLossUW) instance
    """
    loss = MultiHeadLossUW(cfg=cfg, device=device)
    return loss
