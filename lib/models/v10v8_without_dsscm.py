import torch
from torch import tensor
import torch.nn as nn
import sys, os
import math
import sys

sys.path.append(os.getcwd())
# sys.path.append("lib/models")
# sys.path.append("lib/utils")
# sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, Select0, \
    Select1, Select2, Select3, SelectDet, SelectDA, SelectLL, Split
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
# from lib.models.pvt_v2 import pvt_v2_b3, pvt_v2_b1
from lib.models.blocks import Add
from lib.models.FIM import FeatureInteractionModule as fim
from lib.models.DSSCM import DetSegSpatialConsistencyModule as DSSCM
from lib.models.HAFIM import HA_FIM
from lib.models.HAFIM_s import HA_FIM_s
from lib.models.C3 import C3
from lib.models.blocks import ASPP, C2f, LargeKernelBlock, Upsample_PixelShuffle
from lib.models.WWF import BiFPN_Layer
from lib.models.CE import SEAttention
from lib.models.SC import DeformableStripConv as SC
from VMamba.vmamba_backbone import VSSMBackbone
from lib.models.blocks2 import C3_DCN
from lib.models.HF_C2f_DCN import HF_C2f_DCN
# from lib.core.DynamicWeighting import DynamicLossWeighting

# The lane line and the driving area segment branches without share information with each other and without link

# v10v3的基础上修改，主要调整整体通道数，dims [64, 128, 256, 512]-> [48, 96, 192, 384]
# hafim处concat split没必要，直接调整通道进入HAFIM即可
# VSSM未曾缩减通道，DSSCM输出det特征，未被利用
# 💡 SOTA 级全尺度语义金字塔解码器网络配置
Net = [
    [35, 64, 74],  # 输出层索引：[检测头, 行驶区域头, 车道线头]
    [-1, VSSMBackbone, []],
    # 0 - Outputs: [C1(B,48,160,160), C2(B,96,80,80), C3(B,192,40,40), C4(B,384,20,20)]
    [0, Select1, []],  # 1
    [0, Select2, []],  # 2
    [0, Select3, []],  # 3

    # Encoder (FPN/PAN)
    [[1, 2, 3], BiFPN_Layer, [[96, 192, 384], 256]],  # 4
    [-1, Select0, []],  # 5  (Det S8) 256
    [4, Select1, []],  # 6  (Det S16) 256
    [4, Select2, []],  # 7  (Det S32) 256

# 💡 [新增] 检测分支的深度局部特征提取 (克制使用 DCN)

    [5, HF_C2f_DCN, [256, 256, 1]], # 新增引脚号: 8
    # Stride 16 和 32 分辨率小，加上 C3_DCN，精准捕捉形变特征！
    [6, HF_C2f_DCN, [256, 256, 1]], # 新增引脚号: 9
    [7, HF_C2f_DCN, [256, 256, 1]], # 新增引脚号: 10

    # DA Encoder
    [1, Conv, [96, 64, 3, 1]],  # 11
    [-1, C3, [64, 64]],  # 12  (DA S8)
    [2, Conv, [192, 128, 3, 1]],  # 13
    [-1, C2f, [128, 128]],  # 14 (DA S16)
    [3, Conv, [384, 256, 3, 1]],  # 15
    [-1, ASPP, [256, 256]],  # 16 (DA S32)

    # LL Encoder
    [1, Conv, [96, 64, 3, 1]],  # 17
    [-1, SC, [64, 64]],  # 18 (LL S8)
    [2, Conv, [192, 128, 3, 1]],  # 19
    [-1, SC, [128, 128]],  # 20 (LL S16)
    [3, Conv, [384, 256, 3, 1]],  # 21
    [-1, SC, [256, 256]],  # 22 (LL S32)

    # === 特征交互阶段 (FIM) ===
    # FIM @ Stride 8 (重型跨域注意力)
    # [[5, 9, 15], Concat, [1]],  # 20
    # [-1, Conv, [384, 384, 1, 1]],  # 21
    # [-1, Split, [128, 128, 128]],  # 22
    [[8,12,18], HA_FIM, [256, 64, 64, 128, 128, 128, 128, 64, 64]],  # 23

    # FIM @ Stride 16 (轻量级全局反哺)
    # [[6, 11, 17], Concat, [1]],  # 24
    # [-1, Conv, [512, 384, 1, 1]],  # 25
    # [-1, Split, [128, 128, 128]],  # 26
    [[9,14,20], HA_FIM, [256, 128, 128, 128, 128, 128, 256, 128, 128]],  # 24

    # FIM @ Stride 32 (轻量级全局反哺)
    # [[7, 13, 19], Concat, [1]],  # 28
    # [-1, Conv, [768, 384, 1, 1]],  # 29
    # [-1, Split, [128, 128, 128]],  # 30
    [[10,16,22], HA_FIM, [256, 256, 256, 128, 128, 128, 512, 256, 256]],  # 25

    # === 检测头 (Detection Head) ===
    [23, SelectDet, []],  # 26 (Det8) 128
    [24, SelectDet, []],  # 27 (Det16)
    [25, SelectDet, []],  # 28 (Det32)

    [[8, 26], Concat, [1]], # 29
    [-1, HF_C2f_DCN, [384, 128, 2]], #30
    [[9, 27], Concat, [1]], #31
    [-1, HF_C2f_DCN, [512, 256, 2]], #32
    [[10, 28], Concat, [1]], #33
    [-1, HF_C2f_DCN, [768, 512, 3]], #34

    [[30, 32, 34], Detect,
     [1, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],  # 35

    # === 分割多尺度金字塔融合 (Top-Down) ===
    # 1. 行驶区域 (DA) FPN
    [25, SelectDA, []],  # 36 (DA32) 256
    [-1, Upsample, [None, 2, 'nearest']],  # 37 (DA32 -> 16)
    [24, SelectDA, []],  # 38 (DA16) 128
    [[37, 38], Concat, [1]],  # 39 融合
    [-1, Conv, [384, 128, 3, 1]],  # 40 (Fused_DA16)
    [-1, Upsample, [None, 2, 'nearest']],  # 41 (Fused_DA16 -> 8)
    [23, SelectDA, []],  # 42 (DA8)
    [[41, 42], Concat, [1]],  # 43 终极融合
    [-1, Conv, [192, 128, 3, 1]],  # 44 (Fused_DA8)

    # 2. 车道线 (LL) FPN
    [25, SelectLL, []],  # 45 (LL32)
    [-1, Upsample, [None, 2, 'nearest']],  # 46 (LL32 -> 16)
    [24, SelectLL, []],  # 47 (LL16)
    [[46, 47], Concat, [1]],  # 48 融合
    [-1, Conv, [384, 128, 3, 1]],  # 49 (Fused_LL16)
    [-1, Upsample, [None, 2, 'nearest']],  # 50 (Fused_LL16 -> 8)
    [23, SelectLL, []],  # 51 (LL8)
    [[50, 51], Concat, [1]],  # 52 终极融合
    [-1, Conv, [192, 128, 3, 1]],  # 53 (Fused_LL8)

    # === 空间一致性约束 (DSSCM) ===
    # 利用最高分辨率的 Det8 掩膜去修剪已经完全融合好的 Fused_DA8 和 Fused_LL8
    # [[30, 44, 53], DSSCM, [128, 128, 128]],  # 54 -> 返回 [DA_mod, LL_mod]
    # [-1, Select0, []],  # 55 (DA_mod8)
    # [54, Select1, []],  # 56 (LL_mod8)





    # === DA 最终解码输出 ===
    [44, Upsample_PixelShuffle, [128, 128]],  # 54 (->160x160)
    [-1, Conv, [128, 64, 3, 1]],  # 55
    [0, Select0, []],  # 56 (获取 Backbone C1: 160x160)
    [[55, 56], Concat, [1]],  # 57
    [-1, Conv, [112, 64, 3, 1]], # 58
    [-1, SEAttention, [64, 16]],  # 59

    [-1, Upsample_PixelShuffle, [64, 64]],  # 60 (->320x320)
    [-1, Conv, [64, 32, 3, 1]],  # 61
    [-1, Upsample_PixelShuffle, [32, 32]],  # 62 (->640x640)
    [-1, Conv, [32, 16, 3, 1]],  # 63
    [-1, Conv, [16, 3, 3, 1]],  # 64 🎯 (DA 最终多分类输出)

    # === LL 最终解码输出 ===
    [53, Upsample_PixelShuffle, [128, 128]],  # 65 (->160x160)
    [-1, Conv, [128, 64, 3, 1]],  # 66
    [[66, 56], Concat, [1]],  # 67 (复用 C1)
    [-1, Conv, [112, 64, 3, 1]], # 68
    [-1, SEAttention, [64, 16]],  # 69

    [-1, Upsample_PixelShuffle, [64, 64]],  # 70 (->320x320)
    [-1, Conv, [64, 32, 3, 1]],  # 71
    [-1, Upsample_PixelShuffle, [32, 32]],  # 72 (->640x640)
    [-1, Conv, [32, 16, 3, 1]],  # 73
    [-1, Conv, [16, 3, 3, 1]]  # 74 🎯 (LL 最终多分类输出)
]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if isinstance(args, dict):
                block_ = block(**args)
            # elif block is SPPF:
            else:
                block_ = block(*args)
            if block is Detect:
                self.detector_index = i

            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        # print(self.names)

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                self.cuda()
                model_out = self.forward(torch.zeros(1, 3, s, s).cuda())
                detects, _, _ = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors = Detector.anchors.cuda()
            Detector.stride = Detector.stride.cuda()
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale

            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    # def forward(self, x):
    #     cache = []
    #     out = []
    #     det_out = None
    #     Da_fmap = []
    #     LL_fmap = []
    #
    #     ll_feature_map = None
    #     LL_HEAD_FINAL_LAYER_IDX = 74
    #     print('hiahiahiahiahaihaihai')
    #     for i, block in enumerate(self.model):
    #         if block.from_ != -1:
    #             current_input = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]
    #
    #             if i == LL_HEAD_FINAL_LAYER_IDX:
    #                 ll_feature_map = current_input
    #             x = current_input
    #         # print(x.shape)
    # print(block)
    # print(i)
    #         # print(type(x))
    #         # if isinstance(x, torch.Tensor):
    #         #     print(x.shape)
    #         # else:
    #         #     print(len(x))
    #         #     print(x[0].shape)
    #         #     print(x[1].shape)
    #         #     print(x[2].shape)
    #         #     print(x[3].shape)
    #         # print(block)
    #         x = block(x)
    #         # if isinstance(x, torch.Tensor):
    #         #     print(x.shape)
    #
    #         # if isinstance(x, )
    #         if i in self.seg_out_idx:  # save driving area segment result
    #
    #             if i == LL_HEAD_FINAL_LAYER_IDX:
    #                 out.append((x, ll_feature_map))
    #             # m = nn.Sigmoid()
    #             # out.append(m(x))
    #             else:
    #                 out.append(x)
    #         if i == self.detector_index:
    #             det_out = x
    #         cache.append(x if block.index in self.save else None)
    #     out.insert(0, det_out)
    #     return out
    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
                                                                             block.from_]  # calculate concat detect
            # print(x.shape)
            # print(block)`
            # print(i)
            # print(type(x))
            # if isinstance(x, torch.Tensor):
            #     print(x.shape)
            # else:
            #     print(len(x))
            #     print(x[0].shape)
            #     print(x[1].shape)
            #     print(x[2].shape)
            #     print(x[3].shape)
            # print(block)
            x = block(x)
            # if isinstance(x, torch.Tensor):
            #     print(x.shape)

            # if isinstance(x, )
            if i in self.seg_out_idx:  # save driving area segment result
                # m = nn.Sigmoid()
                # out.append(m(x))
                out.append(x)
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0, det_out)
        return out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_net(cfg, **kwargs):
    m_block_cfg = Net
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out, SAD_out = model(input_)
    detects, dring_area_seg, lane_line_seg = model_out
    Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
