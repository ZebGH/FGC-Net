import time
from lib.core.evaluate import ConfusionMatrix, SegmentationMetric
from lib.core.general import non_max_suppression, check_img_size, scale_coords, xyxy2xywh, xywh2xyxy, box_iou, \
    coco80_to_coco91_class, plot_images, ap_per_class, output_to_target
from lib.utils.utils import time_synchronized
from lib.utils import plot_img_and_mask, plot_one_box, show_seg_result
import torch
from threading import Thread
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import json
import random
import cv2
import os
import math
from torch.cuda import amp
from tqdm import tqdm


def train(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup,
          writer_dict, logger, device, rank=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    start = time.time()
    for i, (input, target, paths, shapes) in enumerate(train_loader):
        intermediate = time.time()
        num_iter = i + num_batch * (epoch - 1)

        if num_iter < num_warmup:
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(num_iter, xi,
                                    [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])

        data_time.update(time.time() - start)
        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target

        with amp.autocast(enabled=device.type != 'cpu'):
            outputs = model(input)
            total_loss, head_losses = criterion(outputs, target, shapes, model)

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if rank in [-1, 0]:
            losses.update(total_loss.item(), input.size(0))
            batch_time.update(time.time() - start)

            if i % cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=input.size(0) / batch_time.val,
                    data_time=data_time, loss=losses)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate(epoch, config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, logger=None, device='cpu', rank=-1):
    max_stride = 32
    weights = None

    save_dir = output_dir + os.path.sep + 'visualization'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE]
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS)
    training = False
    is_coco = False
    save_conf = False
    verbose = False
    save_hybrid = False
    log_imgs, wandb = min(16, 100), None

    nc = 1
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    try:
        import wandb
    except ImportError:
        wandb = None
        log_imgs = 0

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=model.nc)

    # === 细粒度（3分类）评测器 ===
    da_metric = SegmentationMetric(3)
    ll_metric = SegmentationMetric(3)

    # === 粗粒度（2分类，对比YOLOP专用）评测器 ===
    da_metric_binary = SegmentationMetric(2)
    ll_metric_binary = SegmentationMetric(2)

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    coco91class = coco80_to_coco91_class()

    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.

    losses = AverageMeter()

    # 行驶区域指标记录器
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()  # 前景 mIoU
    da_mIoU_seg = AverageMeter()  # 包含背景的全局 mIoU
    da_IoU_c1 = AverageMeter()  # 细类 1 (直接行驶区域)
    da_IoU_c2 = AverageMeter()  # 细类 2 (间接行驶区域)
    da_IoU_binary = AverageMeter()  # 二分类 IoU
    da_acc_binary = AverageMeter()  # 💡新增：二分类 Acc
    da_mIoU_binary = AverageMeter()  # 💡新增：二分类 全局mIoU



    # 车道线指标记录器
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()  # 前景 mIoU
    ll_mIoU_seg = AverageMeter()  # 包含背景的全局 mIoU
    ll_IoU_c1 = AverageMeter()  # 细类 1 (实线)
    ll_IoU_c2 = AverageMeter()  # 细类 2 (虚线)
    ll_IoU_binary = AverageMeter()  # 二分类 IoU
    ll_acc_binary = AverageMeter()  # 💡新增：二分类 Acc
    ll_mIoU_binary = AverageMeter()  # 💡新增：二分类 全局mIoU

    T_inf = AverageMeter()
    T_nms = AverageMeter()

    model.eval()
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape

        with torch.no_grad():
            t = time_synchronized()
            det_out, da_seg_out, ll_seg_out = model(img)
            t_inf = time_synchronized() - t
            if batch_i > 0:
                T_inf.update(t_inf / img.size(0), img.size(0))

            inf_out, train_out = det_out

            # 1. 获取预测类别
            _, da_predict = torch.max(da_seg_out, 1)
            _, ll_predict = torch.max(ll_seg_out, 1)

            # 2. 获取标签
            da_gt = target[1]
            ll_gt = target[2]

            # 3. 逐个样本处理
            for i in range(nb):
                pad_w, pad_h = shapes[i][1][1]
                pad_w, pad_h = int(pad_w), int(pad_h)

                da_predict_single = da_predict[i]
                da_gt_single = da_gt[i]
                ll_predict_single = ll_predict[i]
                ll_gt_single = ll_gt[i]

                h, w = da_predict_single.shape

                da_predict_cropped = da_predict_single[pad_h: h - pad_h, pad_w: w - pad_w]
                da_gt_cropped = da_gt_single[pad_h: h - pad_h, pad_w: w - pad_w]
                ll_predict_cropped = ll_predict_single[pad_h: h - pad_h, pad_w: w - pad_w]
                ll_gt_cropped = ll_gt_single[pad_h: h - pad_h, pad_w: w - pad_w]

                # (1) 细粒度（3分类）评估：原封不动传进去
                da_metric.addBatch(da_predict_cropped.cpu(), da_gt_cropped.cpu())
                ll_metric.addBatch(ll_predict_cropped.cpu(), ll_gt_cropped.cpu())

                # (2) 粗粒度（2分类）评估：全部二值化 > 0 即为 1
                da_pred_bin = (da_predict_cropped > 0).long()
                ll_pred_bin = (ll_predict_cropped > 0).long()
                da_gt_bin = (da_gt_cropped > 0).long()
                ll_gt_bin = (ll_gt_cropped > 0).long()

                da_metric_binary.addBatch(da_pred_bin.cpu(), da_gt_bin.cpu())
                ll_metric_binary.addBatch(ll_pred_bin.cpu(), ll_gt_bin.cpu())

            # --- 计算细粒度整个 batch 的指标 ---
            da_acc = da_metric.pixelAccuracy()
            da_ious = da_metric.class_IntersectionOverUnion()

            da_acc_seg.update(da_acc, nb)
            da_IoU_seg.update(da_metric.meanIntersectionOverUnion_fg(), nb)
            da_mIoU_seg.update(da_metric.meanIntersectionOverUnion(), nb)
            if len(da_ious) > 1: da_IoU_c1.update(da_ious[1], nb)
            if len(da_ious) > 2: da_IoU_c2.update(da_ious[2], nb)

            ll_acc = ll_metric.pixelAccuracy()
            ll_ious = ll_metric.class_IntersectionOverUnion()

            ll_acc_seg.update(ll_acc, nb)
            ll_IoU_seg.update(ll_metric.meanIntersectionOverUnion_fg(), nb)
            ll_mIoU_seg.update(ll_metric.meanIntersectionOverUnion(), nb)
            if len(ll_ious) > 1: ll_IoU_c1.update(ll_ious[1], nb)
            if len(ll_ious) > 2: ll_IoU_c2.update(ll_ious[2], nb)

            # --- 计算粗粒度（二分类）整个 batch 的指标 ---
            da_ious_bin = da_metric_binary.class_IntersectionOverUnion()
            da_acc_binary.update(da_metric_binary.pixelAccuracy(), nb)  # 💡新增
            da_mIoU_binary.update(da_metric_binary.meanIntersectionOverUnion(), nb)  # 💡新增
            if len(da_ious_bin) > 1:
                da_IoU_binary.update(da_ious_bin[1], nb)

            ll_ious_bin = ll_metric_binary.class_IntersectionOverUnion()
            ll_acc_binary.update(ll_metric_binary.pixelAccuracy(), nb)  # 💡新增
            ll_mIoU_binary.update(ll_metric_binary.meanIntersectionOverUnion(), nb)  # 💡新增
            if len(ll_ious_bin) > 1:
                ll_IoU_binary.update(ll_ious_bin[1], nb)

            # 重置 metric 对象
            da_metric.reset()
            ll_metric.reset()
            da_metric_binary.reset()
            ll_metric_binary.reset()

            total_loss, head_losses = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model)
            losses.update(total_loss.item(), img.size(0))

            # NMS
            t = time_synchronized()
            target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
            lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
            output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRESHOLD,
                                         iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)

            t_nms = time_synchronized() - t
            if batch_i > 0:
                T_nms.update(t_nms / img.size(0), img.size(0))

            if config.TEST.PLOTS:
                if batch_i < 3:
                    for i in range(nb):
                        img_test = cv2.imread(paths[i])
                        h0, w0 = img_test.shape[:2]
                        ratio = shapes[i][1][0]
                        pad = shapes[i][1][1]

                        # 可行驶区域可视化
                        da_predict_logits = da_seg_out[i]
                        da_gt_single = target[1][i]
                        _, da_predict_map = torch.max(da_predict_logits, 0)

                        h, w = da_predict_map.shape
                        pad_w, pad_h = int(pad[0]), int(pad[1])
                        da_predict_cropped = da_predict_map[pad_h:h - pad_h, pad_w:w - pad_w]
                        da_gt_cropped = da_gt_single[pad_h:h - pad_h, pad_w:w - pad_w]

                        da_predict_vis = cv2.resize(da_predict_cropped.cpu().numpy(), (w0, h0),
                                                    interpolation=cv2.INTER_NEAREST)
                        da_gt_vis = cv2.resize(da_gt_cropped.cpu().numpy(), (w0, h0), interpolation=cv2.INTER_NEAREST)

                        img_da_pred = img_test.copy()
                        img_da_gt = img_test.copy()
                        _ = show_seg_result(img_da_pred, da_predict_vis, i, epoch, save_dir)
                        _ = show_seg_result(img_da_gt, da_gt_vis, i, epoch, save_dir, is_gt=True)

                        # 车道线可视化
                        ll_predict_logits = ll_seg_out[i]
                        ll_gt_single = target[2][i]
                        _, ll_predict_map = torch.max(ll_predict_logits, 0)

                        ll_predict_cropped = ll_predict_map[pad_h:h - pad_h, pad_w:w - pad_w]
                        ll_gt_cropped = ll_gt_single[pad_h:h - pad_h, pad_w:w - pad_w]

                        ll_predict_vis = cv2.resize(ll_predict_cropped.cpu().numpy(), (w0, h0),
                                                    interpolation=cv2.INTER_NEAREST)
                        ll_gt_vis = cv2.resize(ll_gt_cropped.cpu().numpy(), (w0, h0), interpolation=cv2.INTER_NEAREST)

                        img_ll_pred = img_test.copy()
                        img_ll_gt = img_test.copy()
                        _ = show_seg_result(img_ll_pred, ll_predict_vis, i, epoch, save_dir, is_ll=True)
                        _ = show_seg_result(img_ll_gt, ll_gt_vis, i, epoch, save_dir, is_ll=True, is_gt=True)

                        # 检测可视化
                        img_det = cv2.imread(paths[i])
                        img_gt = img_det.copy()
                        det = output[i].clone()
                        if len(det):
                            det[:, :4] = scale_coords(img[i].shape[1:], det[:, :4], img_det.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img_det, label=label_det_pred, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir + "/batch_{}_{}_det_pred.png".format(epoch, i), img_det)

                        labels = target[0][target[0][:, 0] == i, 1:]
                        labels[:, 1:5] = xywh2xyxy(labels[:, 1:5])
                        if len(labels):
                            labels[:, 1:5] = scale_coords(img[i].shape[1:], labels[:, 1:5], img_gt.shape).round()
                        for cls, x1, y1, x2, y2 in labels:
                            label_det_gt = f'{names[int(cls)]}'
                            xyxy = (x1, y1, x2, y2)
                            plot_one_box(xyxy, img_gt, label=label_det_gt, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir + "/batch_{}_{}_det_gt.png".format(epoch, i), img_gt)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = target[0][target[0][:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            if config.TEST.SAVE_TXT:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if config.TEST.PLOTS and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            if config.TEST.SAVE_JSON:
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])
                box[:, :2] -= box[:, 2:] / 2
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                if config.TEST.PLOTS:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                    if pi.shape[0]:
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    map70 = None
    map75 = None
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap70, ap75, ap = ap[:, 0], ap[:, 4], ap[:, 5], ap.mean(1)
        mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)

    pf = '%20s' + '%12.3g' * 6
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    if config.TEST.PLOTS:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

    if config.TEST.SAVE_JSON and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        anno_json = '../coco/annotations/instances_val2017.json'
        pred_json = str(save_dir / f"{w}_predictions.json")
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in val_loader.dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if config.TEST.SAVE_TXT else ''
        print(f"Results saved to {save_dir}{s}")

    model.float()
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    # ====================================================================
    # 最终打印日志：分别展示两套指标，写论文 Baseline 对比必备
    # ====================================================================
    if logger is not None:
        logger.info("\n" + "=" * 65)
        logger.info("--- 1. Coarse-grained Metrics (For YOLOP Comparison) ---")
        logger.info(
            f"[DA] Acc: {da_acc_binary.avg:.4f}  |  IoU: {da_IoU_binary.avg:.4f}  |  mIoU: {da_mIoU_binary.avg:.4f}")
        logger.info(
            f"[LL] Acc: {ll_acc_binary.avg:.4f}  |  IoU: {ll_IoU_binary.avg:.4f}  |  mIoU: {ll_mIoU_binary.avg:.4f}")
        logger.info("-" * 65)
        logger.info("--- 2. Fine-grained Metrics (Our SOTA Contribution) ---")
        logger.info(f"[DA] Acc: {da_acc_seg.avg:.4f}  |  fg-mIoU: {da_IoU_seg.avg:.4f}  |  mIoU: {da_mIoU_seg.avg:.4f}")
        logger.info(f"     ├─ Direct Area IoU:   {da_IoU_c1.avg:.4f}")
        logger.info(f"     └─ Indirect Area IoU: {da_IoU_c2.avg:.4f}")
        logger.info(f"[LL] Acc: {ll_acc_seg.avg:.4f}  |  fg-mIoU: {ll_IoU_seg.avg:.4f}  |  mIoU: {ll_mIoU_seg.avg:.4f}")
        logger.info(f"     ├─ Solid Line IoU:    {ll_IoU_c1.avg:.4f}")
        logger.info(f"     └─ Dashed Line IoU:   {ll_IoU_c2.avg:.4f}")
        logger.info("=" * 65 + "\n")

    # 返回值保持原有结构，确保 train.py 接收不出错
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)

    detect_result = np.asarray([mp, mr, map50, map])
    t = [T_inf.avg, T_nms.avg]
    return da_segment_result, ll_segment_result, detect_result, losses.avg, maps, t


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0