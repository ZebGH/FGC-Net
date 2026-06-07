# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .general import bbox_iou
from .postprocess import build_targets


# --- 新增：OHEM Cross Entropy Loss ---
class OhemCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000):
        super(OhemCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, target):
        # pred: [B, C, H, W], target: [B, H, W]
        # 1. 直接计算完整 Loss (无需 reshape, PyTorch 原生支持 4D)
        loss = self.criterion(pred, target)  # 输出形状: [B, H, W]

        # 2. 展平进行掩码操作
        loss = loss.view(-1)
        target = target.view(-1)
        valid_mask = target != self.ignore_index
        num_valid = valid_mask.sum()

        if self.thresh > 0:
            # 💡 性能黑科技: CE_Loss = -log(Prob) => Prob = exp(-CE_Loss)
            # 这样直接跳过了极度耗时的 Softmax 和 Gather 操作！
            pred_prob = torch.exp(-loss)

            # 找到困难样本 (预测概率小于阈值的)
            mask = pred_prob < self.thresh
            valid_mask = valid_mask & mask

            # 确保至少保留 min_kept 个样本，防止梯度消失
            if valid_mask.sum() < self.min_kept and num_valid > self.min_kept:
                _, indices = loss.topk(self.min_kept)
                valid_mask = torch.zeros_like(target, dtype=torch.bool)
                valid_mask[indices] = True

        return loss[valid_mask].mean() if valid_mask.sum() > 0 else loss.sum() * 0.0


# --- 新增：Multi-Class Dice Loss ---
class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] (logits), targets: [B, H, W]
        B, C, H, W = inputs.shape
        inputs = F.softmax(inputs, dim=1)

        # 1. 过滤 ignore_index 掩码 [B, 1, H, W]
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()

        # 2. 为了不出界，把 ignore_index 替换为 0 (反正一会儿会被 valid_mask 乘掉)
        safe_targets = targets.clone()
        safe_targets[targets == self.ignore_index] = 0

        # 3. 💡 性能黑科技: 用 one_hot 替代 for 循环，全矩阵并行计算！
        # targets_one_hot: [B, C, H, W]
        targets_one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # 4. 在 H, W 维度上求和 (dim=(2, 3))，结果形状 [B, C]
        inter = (inputs * targets_one_hot * valid_mask).sum(dim=(2, 3))
        union = (inputs * valid_mask).sum(dim=(2, 3)) + (targets_one_hot * valid_mask).sum(dim=(2, 3))

        # 5. 计算 Dice
        dice = 1.0 - (2.0 * inter + 1e-4) / (union + 1e-4)

        # 6. 切片去掉背景(索引0)，只对前景类别求平均
        dice_loss = dice[:, 1:].mean()

        return dice_loss


# --- 修改：整合多种 Loss 的 MultiHeadLoss ---
class MultiHeadLoss(nn.Module):
    def __init__(self, losses, cfg, lambdas=None):
        super().__init__()
        self.BCEcls, self.BCEobj = losses[0], losses[1]
        self.OhemCE_da, self.OhemCE_ll = losses[2], losses[3]
        self.Dice_da, self.Dice_ll = losses[4], losses[5]
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)
        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        cfg = self.cfg
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # --- 检测损失部分 (保持你的原样不动) ---
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

        # --- 分割损失部分 (核心修改：联合 OHEM CE 和 Dice Loss) ---
        da_seg_predicts = predictions[1]
        da_seg_targets = targets[1].long()

        ll_seg_predicts = predictions[2]
        ll_seg_targets = targets[2].long()

        # 计算联合 Loss (系数可根据训练情况微调，一般行驶区域 CE 为主，车道线需要更多 Dice 约束)
        lseg_da = self.OhemCE_da(da_seg_predicts, da_seg_targets) + 0.5 * self.Dice_da(da_seg_predicts, da_seg_targets)
        lseg_ll = self.OhemCE_ll(ll_seg_predicts, ll_seg_targets) + 1.0 * self.Dice_ll(ll_seg_predicts, ll_seg_targets)
        # ------------------------------------

        # 占位旧的 liou_ll 保证返回值兼容
        liou_ll = torch.tensor(0.0, device=device)

        s = 3 / no
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]
        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]

        loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll
        return loss, (
        lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), liou_ll.item(), loss.item())


def get_loss(cfg, device):
    """
    get MultiHeadLoss
    """
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)

    # --- 新增：初始化分割损失 ---
    # 假设你的细粒度任务是 3 分类 (背景, 类1, 类2)
    num_seg_classes = 3
    ignore_label = cfg.DATASET.get('IGNORE_LABEL', 255)

    OhemCE_da = OhemCrossEntropyLoss(ignore_index=ignore_label).to(device)
    OhemCE_ll = OhemCrossEntropyLoss(ignore_index=ignore_label).to(device)

    Dice_da = MultiClassDiceLoss(num_classes=num_seg_classes, ignore_index=ignore_label).to(device)
    Dice_ll = MultiClassDiceLoss(num_classes=num_seg_classes, ignore_index=ignore_label).to(device)

    # Focal loss (只对 BCE 损失有效)
    gamma = cfg.LOSS.FL_GAMMA
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    # 将所有损失函数传入 MultiHeadLoss
    loss_list = [BCEcls, BCEobj, OhemCE_da, OhemCE_ll, Dice_da, Dice_ll]

    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss