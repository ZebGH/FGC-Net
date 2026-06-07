# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import general


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='precision-recall_curve.png', names=[]):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls)

    px, py = np.linspace(0, 1, 1000), []
    pr_score = 0.1
    s = [unique_classes.shape[0], tp.shape[1]]
    ap, p, r = np.zeros(s), np.zeros((unique_classes.shape[0], 1000)), np.zeros((unique_classes.shape[0], 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_l == 0:
            continue
        else:
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            recall = tpc / (n_l + 1e-16)
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

            precision = tpc / (tpc + fpc)
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and (j == 0):
                    py.append(np.interp(px, mrec, mpre))

    f1 = 2 * p * r / (p + r + 1e-16)
    i = r.mean(0).argmax()

    if plot:
        plot_pr_curve(px, py, ap, save_dir, names)

    return p[:, i], r[:, i], ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    method = 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, mpre, mrec


class ConfusionMatrix:
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1
            else:
                self.matrix[gc, self.nc] += 1

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[self.nc, dc] += 1

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn
            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)
            array[array < 0.005] = np.nan
            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)
            labels = (0 < len(names) < 99) and len(names) == self.nc
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FN'] if labels else "auto",
                       yticklabels=names + ['background FP'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


class SegmentationMetric(object):
    '''
    支持多分类与二分类的统一评估指标库
    '''
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        # 整体像素准确率
        acc = np.diag(self.confusionMatrix).sum() / (self.confusionMatrix.sum() + 1e-12)
        return acc

    def classPixelAccuracy(self):
        # 每一类的像素准确率
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-12)
        return classAcc

    def class_IntersectionOverUnion(self):
        # 返回所有类别的 IoU 数组 [IoU_bg, IoU_class1, IoU_class2, ...]
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / (union + 1e-12)
        IoU[np.isnan(IoU)] = 0
        return IoU

    def meanIntersectionOverUnion(self):
        # 传统的 mIoU (包含背景)
        IoU = self.class_IntersectionOverUnion()
        mIoU = np.nanmean(IoU)
        return mIoU

    def meanIntersectionOverUnion_fg(self):
        # 前景 mIoU (排除背景索引 0，只看目标类别的平均表现)
        IoU = self.class_IntersectionOverUnion()
        if self.numClass > 1:
            mIoU_fg = np.nanmean(IoU[1:])
        else:
            mIoU_fg = IoU[0]
        return mIoU_fg

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # 过滤 ignore_label
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='.', names=()):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} %.3f' % ap[i, 0])
    else:
        ax.plot(px, py, linewidth=1, color='grey')

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir) / 'precision_recall_curve.png', dpi=250)