# -*- coding: utf-8 -*-

import numpy as np
import cv2
import random
import math


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def random_perspective(combination_tuple, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0),
                       img_interp_flag=cv2.INTER_LINEAR, mask_interp_flag=cv2.INTER_NEAREST,
                       img_border_value=(114, 114, 114), mask_border_values=(0, 0)):  # (da_border_val, ll_border_val)
    """Applies random perspective and affine transformations to image and masks.
       combination_tuple: (img, da_mask, ll_mask)
       mask_border_values: A tuple (da_mask_border_value, ll_mask_border_value)
    """
    img, da_mask, ll_mask = combination_tuple  # Unpack

    height_orig = img.shape[0] + border[0] * 2
    width_orig = img.shape[1] + border[1] * 2

    # Center (using original image dimensions before potential border addition by this function)
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective, Rotation, Scale, Shear, Translation matrices (P, R, S, T)
    # ... (Keep the matrix calculations M = T @ S @ R @ P @ C as is) ...
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width_orig  # Use width_orig for translate reference
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height_orig  # Use height_orig for translate reference
    M = T @ S @ R @ P @ C

    img_out = img
    da_mask_out = da_mask
    ll_mask_out = ll_mask

    # Effective dsize for warp, including any border added by this function
    dsize_warp = (width_orig, height_orig)  # warp to original dims before specific border
    if border[0] != 0 or border[1] != 0:  # if border is added by this function, use effective height/width
        dsize_warp = (height_orig + border[0] * 2, width_orig + border[1] * 2)

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img_out = cv2.warpPerspective(img, M, dsize=dsize_warp, flags=img_interp_flag,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=img_border_value)
            da_mask_out = cv2.warpPerspective(da_mask, M, dsize=dsize_warp, flags=mask_interp_flag,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=mask_border_values[0])
            ll_mask_out = cv2.warpPerspective(ll_mask, M, dsize=dsize_warp, flags=mask_interp_flag,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=mask_border_values[1])
        else:  # affine
            img_out = cv2.warpAffine(img, M[:2], dsize=dsize_warp, flags=img_interp_flag,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=img_border_value)
            da_mask_out = cv2.warpAffine(da_mask, M[:2], dsize=dsize_warp, flags=mask_interp_flag,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=mask_border_values[0])
            ll_mask_out = cv2.warpAffine(ll_mask, M[:2], dsize=dsize_warp, flags=mask_interp_flag,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=mask_border_values[1])

    # ... (Transform label coordinates 'targets' logic remains largely the same) ...
    # Ensure 'width' and 'height' used for clipping targets match dsize_warp
    final_width_for_clip, final_height_for_clip = dsize_warp

    n = len(targets)
    targets_out = targets.copy()  # Ensure we modify a copy
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        else:
            xy = xy[:, :2].reshape(n, 8)

        x_coords = xy[:, [0, 2, 4, 6]]
        y_coords = xy[:, [1, 3, 5, 7]]
        xy_transformed = np.concatenate((x_coords.min(1), y_coords.min(1), x_coords.max(1), y_coords.max(1))).reshape(4,
                                                                                                                      n).T

        xy_transformed[:, [0, 2]] = xy_transformed[:, [0, 2]].clip(0,
                                                                   final_width_for_clip)  # Use dsize_warp for clipping
        xy_transformed[:, [1, 3]] = xy_transformed[:, [1, 3]].clip(0,
                                                                   final_height_for_clip)  # Use dsize_warp for clipping

        # filter candidates (re-evaluate s for box_candidates if necessary)
        # The original 's' (scale factor in Rotation matrix) might not be the only scale affecting targets
        # For simplicity, assuming box_candidates handles varying scales or relative sizes.
        i = _box_candidates(box1=targets[:, 1:5].T * s,
                            box2=xy_transformed.T)  # `s` here refers to the scale param of R matrix
        targets_out = targets[i]
        if len(targets_out) > 0:  # only assign if there are filtered targets
            targets_out[:, 1:5] = xy_transformed[i]
        else:  # if all targets are filtered out
            targets_out = np.empty((0, targets.shape[1])) if isinstance(targets, np.ndarray) else []

    combination_out = (img_out, da_mask_out, ll_mask_out)
    return combination_out, targets_out


def cutout(combination, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    image, gray = combination
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)
        # print('xmin:{},ymin:{},xmax:{},ymax:{}'.format(xmin,ymin,xmax,ymax))

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
        gray[ymin:ymax, xmin:xmax] = -1

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return image, gray, labels


def letterbox(img_masks_tuple, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
              img_interp=cv2.INTER_LINEAR, mask_interp=cv2.INTER_NEAREST,
              mask_pad_values=(0, 0)):  # Expects (da_pad_value, ll_pad_value)
    """Resizes and pads image/masks while meeting stride-multiple constraints.
       img_masks_tuple: (img, da_mask, ll_mask)
       mask_pad_values: A tuple (da_mask_pad_value, ll_mask_pad_value)
    """
    img, da_mask, ll_mask = img_masks_tuple
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    # Resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=img_interp)
        da_mask = cv2.resize(da_mask, new_unpad, interpolation=mask_interp)  # NEAREST for DA mask
        ll_mask = cv2.resize(ll_mask, new_unpad, interpolation=mask_interp)  # NEAREST for LL mask

    # Padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    da_mask = cv2.copyMakeBorder(da_mask, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=mask_pad_values[0])  # DA pad value
    ll_mask = cv2.copyMakeBorder(ll_mask, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=mask_pad_values[1])  # LL pad value (usually 0)

    return (img, da_mask, ll_mask), ratio, (dw, dh)


def letterbox_for_img(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def _box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates
