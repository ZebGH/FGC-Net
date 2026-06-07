# loss.py

import torch.nn as nn
import torch
from .general import bbox_iou
from .postprocess import build_targets


# 移除 SegmentationMetric，因为旧的 liou_ll 计算方式不可微且不适用于此
# from lib.core.evaluate import SegmentationMetric

class MultiHeadLoss(nn.Module):
    def __init__(self, losses, cfg, lambdas=None):
        super().__init__()
        # ... ( lambdas 的处理逻辑可以保持，或者简化，因为我们将使用新的损失) ...
        # self.losses 现在将包含 BCEcls, BCEobj, 和我们新定义的 CELoss
        self.BCEcls, self.BCEobj, self.CELoss = losses[0], losses[1], losses[2]
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)
        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        cfg = self.cfg
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # --- 检测损失部分 (保持不变) ---
        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)
        cp, cn = smooth_BCE(eps=0.0)
        nt = 0
        no = len(predictions[0])
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]
        for i, pi in enumerate(predictions[0]):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            n = b.shape[0]
            if n:
                nt += n
                ps = pi[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)
                if model.nc > 1:
                    t = torch.full_like(ps[:, 5:], cn, device=device)
                    t[range(n), tcls[i]] = cp
                    lcls += self.BCEcls(ps[:, 5:], t)
            lobj += self.BCEobj(pi[..., 4], tobj) * balance[i]
        # ------------------------------------

        # --- 分割损失部分 (核心修改) ---
        # 原代码:
        # drive_area_seg_predicts = predictions[1].view(-1)
        # drive_area_seg_targets = targets[1].view(-1)
        # lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)
        # lane_line_seg_predicts = predictions[2].view(-1)
        # lane_line_seg_targets = targets[2].view(-1)
        # lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)

        # 新代码:
        # CrossEntropyLoss 需要模型输出 (logits) 和标签 (target)
        # 模型输出 predictions[1] 的形状是 [B, 3, H, W]
        # 标签 targets[1] 的形状是 [B, H, W]，类型是 long
        da_seg_predicts = predictions[1]
        da_seg_targets = targets[1]

        da_seg_targets = da_seg_targets.long()

        lseg_da = self.CELoss(da_seg_predicts, da_seg_targets)

        ll_seg_predicts = predictions[2]
        ll_seg_targets = targets[2]

        ll_seg_targets = ll_seg_targets.long()

        lseg_ll = self.CELoss(ll_seg_predicts, ll_seg_targets)
        # ------------------------------------

        # --- 移除旧的、不可微的 IoU loss 计算 ---
        # liou_ll 的计算方式依赖于 torch.max 和 .cpu()，这会中断计算图，无法用于训练
        # 所以我们直接移除它
        liou_ll = torch.tensor(0.0, device=device)  # 将其设为0
        # ------------------------------------

        s = 3 / no
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]
        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
        # liou_ll 的权重也应处理，但因为它现在是0，所以无影响
        # liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[5]

        # ... (条件训练 if cfg.TRAIN.DET_ONLY 等逻辑保持不变) ...

        loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll  # liou_ll 现在是0
        return loss, (
        lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), liou_ll.item(), loss.item())


def get_loss(cfg, device):
    """
    get MultiHeadLoss
    """
    # --- 修改损失函数的定义 ---
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)

    # 原代码:
    # segmentation loss criteria
    # BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)

    # 新代码:
    # segmentation loss criteria for multi-class
    # 我们可以为不同类别设置权重，例如，如果某个车道线类别很少，可以给它更高的权重
    # ignore_index=255 (或您cfg中定义的值) 可以让损失函数忽略掉标签中像素值为255的区域
    CELoss = nn.CrossEntropyLoss(ignore_index=cfg.DATASET.get('IGNORE_LABEL', 255)).to(device)

    # Focal loss (只对 BCE 损失有效)
    gamma = cfg.LOSS.FL_GAMMA
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    # 原代码: loss_list = [BCEcls, BCEobj, BCEseg]
    # 新代码:
    loss_list = [BCEcls, BCEobj, CELoss]  # 将 CELoss 放入列表

    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss

# ... (smooth_BCE 和 FocalLoss 类保持不变) ...

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


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