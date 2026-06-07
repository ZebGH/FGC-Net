import argparse
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np

# 导入项目自带的模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from lib.utils import DataLoaderX
import lib.dataset as dataset
from lib.config import cfg, update_config
from lib.core.celoss import get_loss
from lib.core.function import validate
from lib.models import get_net
from lib.utils.utils import create_logger, select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained Multitask network')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights (.pth)')
    parser.add_argument('--logDir', type=str, default='runs/eval_logs/', help='log directory')
    # 兼容 YOLOP 原有的 cfg 等参数
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    # 1. 初始化环境与 Logger
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.logDir, 'eval')
    logger.info(f"==> Evaluating weights from: {args.weights}")

    device = select_device(logger, '0')  # 默认使用 GPU 0 进行评估
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # 2. 构建模型并加载权重
    logger.info("==> Building model...")
    model = get_net(cfg).to(device)

    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        # 兼容两种保存格式 (带 optimizer 状态的 checkpoint 和纯净的 state_dict)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"==> Successfully loaded weights from {args.weights}")
    else:
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    # 如果有多个 GPU，可以包裹 DataParallel（测试一般单卡就够了）
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 3. 构建验证集 DataLoader
    logger.info("==> Loading Validation Dataset...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS) if len(cfg.GPUS) > 0 else cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )

    # 4. 获取 Loss (即使是评估，YOLOP 的 validate 逻辑也需要 criterion 计算 valid loss)
    criterion = get_loss(cfg, device=device)

    # 构建空的 writer_dict，测试不需要 Tensorboard
    writer_dict = {'writer': None, 'train_global_steps': 0, 'valid_global_steps': 0}

    # 5. 开始评估
    logger.info("==> Start validation...")
    model.eval()

    with torch.no_grad():
        da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
            epoch=0,
            config=cfg,
            val_loader=valid_loader,
            val_dataset=valid_dataset,
            model=model,
            criterion=criterion,
            output_dir=final_output_dir,
            tb_log_dir=tb_log_dir,
            writer_dict=writer_dict,
            logger=logger,
            device=device,
            rank=-1
        )

    logger.info("==> Evaluation Completed!")


if __name__ == '__main__':
    main()