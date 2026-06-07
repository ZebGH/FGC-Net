import os
from yacs.config import CfgNode as CN

_C = CN()

_C.LOG_DIR = 'runs/'
_C.GPUS = [0]
_C.WORKERS = 16
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False   # Resume from the last training interrupt
_C.NEED_AUTOANCHOR = False  # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
_C.DEBUG = False
_C.num_seg_class = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.STRU_WITHSHARE = False  # add share_block to segbranch
_C.MODEL.HEADS_NAME = ['']
_C.MODEL.PRETRAINED = ""
_C.MODEL.PRETRAINED_DET = ""
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)

# 在 default.py 文件中
# ... 其他配置 ...


# ... 其他 MODEL 设置 ...
# _C.MODEL.NUM_DA_CLASSES = 3  # 可行驶区域类别数 (0: direct, 1: alternative, 2: background)
# _C.MODEL.NUM_LL_CLASSES = 9  # 车道线类别数 (0: background, 1-8: 对应您定义的8种车道线类型 - 这个数字需要根据您最终的 LANE_TYPE_TO_ID 确定！)

# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
# _C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.MULTI_HEAD_LAMBDA = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.BOX_GAIN = 0.2  # box loss gain
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 0.7  # object loss gain
_C.LOSS.DA_SEG_GAIN = 0.2  # driving area segmentation loss gain
_C.LOSS.LL_SEG_GAIN = 0.25  # lane line segmentation loss gain
_C.LOSS.LL_IOU_GAIN = 0.3  # lane line iou loss gain

_C.LOSS.SEG_LOSS_ALPHA = 0.5  # 控制 CE 和 Dice Loss 的权重 (例如 0.5*CE + 0.5*Dice) - 可以调整

# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = 'D://PythonProjects/ZebProjects/TrissFormer/data/fine/images/'  # the path of images folder
_C.DATASET.LABELROOT = 'D://PythonProjects/ZebProjects/TrissFormer/data/fine/tp_anno/'  # the path of det_annotations folder
_C.DATASET.MASKROOT = 'D://PythonProjects/ZebProjects/TrissFormer/data/fine/da_anno/'  # the path of da_seg_annotations folder
_C.DATASET.LANEROOT = 'D://PythonProjects/ZebProjects/TrissFormer/data/fine/ll_anno/'  # the path of ll_seg_annotations folder
_C.DATASET.DATASET = 'BddDataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [720, 1280]
_C.DATASET.IGNORE_LABEL = 255  # 或者 BDD100k 通常用的背景/忽略标签值 (如果 DA/LL 的背景类不参与损失计算或评估)

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add


# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.0002  # initial learning rate (SGD=1E-2, Adam=1E-3) 0.000125
_C.TRAIN.LRF = 0.05  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 5.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.05
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 240

_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE_PER_GPU = 22
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

# if training 3 tasks end-to-end, set all parameters as True
# Alternating optimization
_C.TRAIN.SEG_ONLY = False  # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False  # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False  # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False  # Only train encoder and detection branch

# Single task
_C.TRAIN.DRIVABLE_ONLY = False  # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False  # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False  # Only train detection task

_C.TRAIN.PLOT = True  #

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 22
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = True
_C.TEST.NMS_CONF_THRESHOLD = 0.001  # 0.001
_C.TEST.NMS_IOU_THRESHOLD = 0.6  # 0.6


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    # if args.conf_thres:
    #     cfg.TEST.NMS_CONF_THRESHOLD = args.conf_thres

    # if args.iou_thres:
    #     cfg.TEST.NMS_IOU_THRESHOLD = args.iou_thres

    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED`
    # )
    #
    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()
