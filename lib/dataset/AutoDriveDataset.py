import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout


class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """

    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize

        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator
        # self.label_list = self.label_root.iterdir()
        self.mask_list = list(self.mask_root.iterdir())

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError

    def __len__(self, ):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise FileNotFoundError(f"Image not found or empty: {data['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- 1. 加载新的 DA 和 LL Mask (作为单通道灰度图) ---
        # data["mask"] 指向 DA mask, data["lane"] 指向 LL mask
        da_label = cv2.imread(data["mask"], cv2.IMREAD_GRAYSCALE)
        if da_label is None:
            raise FileNotFoundError(f"DA mask not found or empty: {data['mask']}")

        ll_label = cv2.imread(data["lane"], cv2.IMREAD_GRAYSCALE)
        if ll_label is None:
            raise FileNotFoundError(f"LL mask not found or empty: {data['lane']}")
        # ----------------------------------------------------

        # --- 2. Resize (对掩码使用最近邻插值) ---
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # 原始尺寸
        r = resized_shape / max(h0, w0)

        img_resized, da_label_resized, ll_label_resized = img, da_label, ll_label
        if r != 1:
            img_resized = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
            da_label_resized = cv2.resize(da_label, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_NEAREST)
            ll_label_resized = cv2.resize(ll_label, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_NEAREST)
        h, w = img_resized.shape[:2]

        # --- 3. Letterbox (调用已修改的函数) ---
        # 确保你 lib/utils/augmentations.py 中的 letterbox 函数是我们之前讨论的修改后版本
        # DA 背景/忽略ID. 如果DA是3分类(0,1,2), 背景通常是2.
        da_pad_val = self.cfg.DATASET.get('DA_IGNORE_LABEL', 2)
        ll_pad_val = 0  # LL 背景ID是0
        (img_letter, da_label_letter, ll_label_letter), ratio, pad = letterbox(
            (img_resized, da_label_resized, ll_label_resized),
            resized_shape, auto=True, scaleup=self.is_train,
            mask_pad_values=(da_pad_val, ll_pad_val)
        )
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # --- 4. 处理检测标签 (保持不变) ---
        det_label = data["label"]
        labels = []
        if det_label.size > 0:
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]

        # --- 5. 数据增强 (训练时) ---
        img_aug, da_label_aug, ll_label_aug = img_letter, da_label_letter, ll_label_letter
        labels_aug = labels

        if self.is_train:
            # 调用已修改的 random_perspective
            (img_aug, da_label_aug, ll_label_aug), labels_aug = random_perspective(
                combination_tuple=(img_letter, da_label_letter, ll_label_letter),
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR,
                perspective=self.cfg.DATASET.get('PERSPECTIVE', 0.0),
                mask_border_values=(da_pad_val, ll_pad_val)
            )

            augment_hsv(img_aug, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S,
                        vgain=self.cfg.DATASET.HSV_V)

            if len(labels_aug):
                labels_aug[:, 1:5] = xyxy2xywh(labels_aug[:, 1:5])
                labels_aug[:, [2, 4]] /= img_aug.shape[0]
                labels_aug[:, [1, 3]] /= img_aug.shape[1]

            if self.flip and random.random() < 0.5:
                img_aug = np.fliplr(img_aug)
                da_label_aug = np.fliplr(da_label_aug)
                ll_label_aug = np.fliplr(ll_label_aug)
                if len(labels_aug):
                    labels_aug[:, 1] = 1 - labels_aug[:, 1]
            # ... ud_flip ...
        else:
            if len(labels_aug):
                labels_aug[:, 1:5] = xyxy2xywh(labels_aug[:, 1:5])
                labels_aug[:, [2, 4]] /= img_aug.shape[0]
                labels_aug[:, [1, 3]] /= img_aug.shape[1]

        # --- 6. 准备最终输出 ---
        labels_out = torch.zeros((len(labels_aug), 6))
        if len(labels_aug):
            labels_out[:, 1:] = torch.from_numpy(labels_aug)

        img_tensor = self.transform(img_aug.copy())

        # 将最终的 NumPy Mask 数组转换为 Long Tensor
        # np.ascontiguousarray 确保内存布局是连续的，有助于提高性能
        da_label_tensor = torch.from_numpy(np.ascontiguousarray(da_label_aug)).long()
        ll_label_tensor = torch.from_numpy(np.ascontiguousarray(ll_label_aug)).long()

        target = [labels_out, da_label_tensor, ll_label_tensor]

        return img_tensor, target, data["image"], shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes = zip(*batch)
        label_det, label_seg, label_lane = [], [], []
        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l
            if l_det.shape[0] > 0:  # 仅当存在检测目标时才添加
                l_det[:, 0] = i
                label_det.append(l_det)

            label_seg.append(l_seg)
            label_lane.append(l_lane)

        # 如果整个批次都没有检测目标，创建一个空的占位符
        if not label_det:
            label_det_out = torch.empty(0, 6)
        else:
            label_det_out = torch.cat(label_det, 0)

        return torch.stack(img, 0), [label_det_out, torch.stack(label_seg, 0),
                                     torch.stack(label_lane, 0)], paths, shapes