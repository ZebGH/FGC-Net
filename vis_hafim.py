import torch
import cv2
import os
import numpy as np
import torchvision.transforms as transforms
from lib.models import get_net
from lib.config import cfg

# ==================== ⚙️ 手动配置区域 ⚙️ ====================

WEIGHT_PATH = '/home/zeb/PythonProjects/FGC-Net/runs/Bdd100k-fg/_2026-03-12-23-00/epoch-240.pth'
IMAGE_PATH = '/home/zeb/data/bdd100kfg/images/val/bd92c756-d4b30972.jpg'  # 


TARGET_INDEX = 0
image_name = IMAGE_PATH.split('/')[-1].split('.jpg')[0] + '_'+str(TARGET_INDEX)

SAVE_NAME = f'./EXP/Ablation/HAFIM/{image_name}'
# ============================================================

activations = {'before': None, 'after': None}


# 获取施法前的检测特征
def hook_before(module, input, output):
    activations['before'] = output.detach()


# 获取施法后的检测特征 (挂在 dist_det 的输入上)
def hook_after(module, input):
    activations['after'] = input[0].detach()


def process_feature_map(feature_tensor, img_shape):
    heatmap = torch.mean(torch.abs(feature_tensor.squeeze(0)), dim=0).cpu().numpy()
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap_resized = cv2.resize(heatmap, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    return heatmap_color


def main():
    print(f"🚀 启动 HAFIM 手术刀... 目标锁定为第 {TARGET_INDEX} 个模块")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_net(cfg).to(device)
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    hafim_modules = []

    # 扫描收集所有的 HA_FIM 和 HA_FIM_s
    for name, m in model.named_modules():
        if m.__class__.__name__ in ['HA_FIM', 'HA_FIM_s']:
            hafim_modules.append((name, m))

    total_hafim = len(hafim_modules)
    if total_hafim == 0:
        print("❌ 没找到任何 HAFIM 模块！")
        return

    print(f"✅ 扫描完毕！共发现 {total_hafim} 个 HAFIM 模块：")
    for i, (name, m) in enumerate(hafim_modules):
        indicator = "👈 [当前选中]" if i == TARGET_INDEX else ""
        print(f"   [{i}] 类型: {m.__class__.__name__} | 路径: {name} {indicator}")

    if TARGET_INDEX >= total_hafim or TARGET_INDEX < 0:
        print(f"❌ 越界了！合法范围是 0 到 {total_hafim - 1}。")
        return

    target_name, target_module = hafim_modules[TARGET_INDEX]

    try:
        # 下刀 1：获取施法前
        target_module.det_enh_proj.register_forward_hook(hook_before)
        # 下刀 2：获取施法后（使用 pre_hook 拦截输入）
        target_module.dist_det.register_forward_pre_hook(hook_after)
        print("✅ 探针插入成功！")
    except Exception as e:
        print(f"❌ 挂载子模块失败: {e}")
        return

    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_img = transforms.ToTensor()(img_resized)
    tensor_img = normalize(tensor_img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(tensor_img)

    if activations['before'] is None or activations['after'] is None:
        print("❌ 特征提取失败！")
        return

    heatmap_before = process_feature_map(activations['before'], img_bgr.shape)
    heatmap_after = process_feature_map(activations['after'], img_bgr.shape)

    result_before = heatmap_before * 0.6 + img_bgr * 0.4
    result_after = heatmap_after * 0.6 + img_bgr * 0.4

    # spacer = np.zeros((img_bgr.shape[0], 10, 3), dtype=np.uint8) + 255
    # concat_img = np.hstack((img_bgr, spacer, result_before, spacer, result_after))

    # cv2.putText(concat_img, f"HAFIM: {target_module.__class__.__name__}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
    #             (255, 255, 255), 2)
    # cv2.putText(concat_img, "Before HAFIM (Det)", (img_bgr.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
    #             (255, 255, 255), 2)
    # cv2.putText(concat_img, "After HAFIM (Det)", (img_bgr.shape[1] * 2 + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
    #             (255, 255, 255), 2)

    cv2.imwrite(f'{SAVE_NAME}_before.jpg', result_before)
    cv2.imwrite(f'{SAVE_NAME}_after.jpg', result_after)
    # print(f"🎉🎉🎉 净化对比图已保存至: {os.path.abspath(SAVE_NAME)}")


if __name__ == '__main__':
    main()