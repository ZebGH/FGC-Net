import argparse
import os, sys
import time
import torch
from thop import profile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torchvision.transforms as transforms
from lib.config import cfg
from lib.utils.utils import select_device
from lib.models import get_net
from lib.dataset import LoadImages
from lib.core.general import non_max_suppression

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def run_benchmark(cfg, opt):
    print("=" * 50)
    print("🚀 正在初始化 FGC-Net Benchmark 环境...")
    device = select_device('', opt.device)

    # 1. 加载模型与权重
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # ==========================================
    # 模块 A: 计算 Params 和 GFLOPs (使用 dummy 张量)
    # ==========================================
    print("\n[1/3] 正在使用 thop 计算 Params 和 GFLOPs...")
    dummy_input = torch.randn(1, 3, opt.img_size, opt.img_size).to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = (macs * 2) / 1e9
    params_m = params / 1e6
    print(f"✅ Params: {params_m:.2f} M")
    print(f"✅ GFLOPs: {gflops:.2f}")

    # ==========================================
    # 模块 B: 准备真实图像并移动到显存
    # ==========================================
    print(f"\n[2/3] 从数据源加载真实图像用于测速 (Source: {opt.source})")
    dataset = LoadImages(opt.source, img_size=opt.img_size)

    # 抓取第一张真实图片并预处理
    iterator = iter(dataset)
    _, img_ori, _, _, _ = next(iterator)
    img = transform(img_ori).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    print(f"✅ 成功加载图像，输入张量维度: {img.shape}")

    # ==========================================
    # 模块 C: 纯净 FPS 测速 (分离 Forward 和 NMS)
    # ==========================================
    print("\n[3/3] 开始 FPS 测试 (NMS Conf=0.24, IoU=0.45)")

    starter_fwd, ender_fwd = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter_nms, ender_nms = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    iterations = 500
    fwd_times = torch.zeros((iterations, 1))
    nms_times = torch.zeros((iterations, 1))

    # --- 预热阶段 (Warm-up) ---
    print("   -> 正在进行 GPU 预热 (50 次)...")
    with torch.no_grad():
        for _ in range(500):
            det_out, da_seg_out, ll_seg_out = model(img)
            inf_out, _ = det_out
            _ = non_max_suppression(inf_out, conf_thres=0.24, iou_thres=0.45, classes=None, agnostic=False)

    # --- 正式测速阶段 ---
    print(f"   -> 开始正式测速 ({iterations} 次循环)...")
    with torch.no_grad():
        for i in range(iterations):
            # 1. 记录前向传播时间
            starter_fwd.record()
            det_out, da_seg_out, ll_seg_out = model(img)
            inf_out, _ = det_out
            ender_fwd.record()
            torch.cuda.synchronize()
            fwd_times[i] = starter_fwd.elapsed_time(ender_fwd)

            # 2. 记录 NMS 时间
            starter_nms.record()
            _ = non_max_suppression(inf_out, conf_thres=0.24, iou_thres=0.45, classes=None, agnostic=False)
            ender_nms.record()
            torch.cuda.synchronize()
            nms_times[i] = starter_nms.elapsed_time(ender_nms)

    # --- 统计与打印 ---
    mean_fwd = torch.mean(fwd_times).item()
    mean_nms = torch.mean(nms_times).item()
    total_time = mean_fwd + mean_nms
    fps = 1000.0 / total_time

    print("\n" + "=" * 50)
    print(f"🎉 评测完成! 模型: FGC-Net (细粒度)")
    print("=" * 50)
    print(f"📌 Input Size: {opt.img_size}x{opt.img_size}")
    print(f"📌 Params:     {params_m:.2f} M")
    print(f"📌 GFLOPs:     {gflops:.2f}")
    print("-" * 50)
    print(f"⏱️ Forward:     {mean_fwd:.2f} ms")
    print(f"⏱️ NMS:         {mean_nms:.2f} ms")
    print(f"⏱️ Total:       {total_time:.2f} ms")
    print(f"🔥 End-to-End FPS: {fps:.2f}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认加载你的权重路径
    parser.add_argument('--weights', type=str, default='F://EXP/5090/v10v9_coarse/epoch-240.pth',
                        help='model.pth path')
    # 数据源只要指向你现有的视频或图片文件夹即可，脚本只提取第一帧用于测速 F:\EXP\5090\v10v8\fgc_fine\runs\BddDataset\_2026-04-09-05-51 F://EXP/5090/v10v10/runs/_2026-04-05-02-40/epoch-240.pth
    parser.add_argument('--source', type=str,
                        default='D://MTK/data/fine/images/val/b1d0a191-06deb55d.jpg', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    opt = parser.parse_args()
    run_benchmark(cfg, opt)
    # 'F://EXP/5090/v10v10/runs/_2026-04-05-02-40/epoch-240.pth'
    # 'F://EXP/5090/v10v8/fgc_fine/runs/BddDataset/_2026-04-09-05-51/epoch-214.pth'
    # 'F://EXP/5090/v10v8/vssm_only/epoch-222.pth'
    # 'F://EXP/5090/v10v5/_2026-03-10-02-29/epoch-240.pth'