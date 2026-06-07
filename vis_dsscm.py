import os
import cv2
import torch
import numpy as np
import types
import torchvision.transforms as transforms

# =========================================================
# ️ 1. 导入网络构建函数
# =========================================================
try:
    from lib.models.fgcnet import get_net

except ImportError:
    print("⚠️ 导入失败，请确保从正确路径导入 get_net 函数！")

# =========================================================
#  2. 截获 DSSCM 特征
# =========================================================
DSSCM_FEATURES = {}


def hook_dsscm_forward(dsscm_module):
    """动态替换 DSSCM 的 forward，把内部手术过程偷出来"""
    old_forward = dsscm_module.forward

    def new_forward(self, x):
        d_prime, s_da_prime, s_ll_prime = x[0], x[1], x[2]

        # 1. 拦截：未雕刻前的 DA 原始特征 (多通道)
        DSSCM_FEATURES['s_da_prime'] = s_da_prime.detach().cpu()

        # 2. 拦截：手术刀 —— 车辆遮挡掩膜 (0~1单通道)
        det_presence_map = self.presence_head(d_prime)
        DSSCM_FEATURES['det_presence_map'] = det_presence_map.detach().cpu()

        # 中间调制运算
        s_da_modulated_raw = s_da_prime * (1.0 - self.alpha * det_presence_map)
        s_ll_modulated_raw = s_ll_prime * (1.0 - self.beta * det_presence_map)

        # 3. 拦截：模具 —— 车道线引导掩膜 (0~1单通道)
        ll_guide_map = self.ll_guide_head(s_ll_modulated_raw)
        DSSCM_FEATURES['ll_guide_map'] = ll_guide_map.detach().cpu()

        # 融合引导
        s_da_modulated_raw = s_da_modulated_raw + self.gamma * (s_da_modulated_raw * ll_guide_map)

        # 4. 拦截：雕刻后的最终 DA 特征
        s_da_modulated_final = self.refine_da(s_da_modulated_raw)
        DSSCM_FEATURES['s_da_modulated_final'] = s_da_modulated_final.detach().cpu()

        # 返回正常结果，不影响后续运算
        s_ll_modulated_final = self.refine_ll(s_ll_modulated_raw)
        return s_da_modulated_final, s_ll_modulated_final

    dsscm_module.forward = types.MethodType(new_forward, dsscm_module)
    return old_forward


# =========================================================
#  3. 绘图引擎：热力图转化与横向拼接
# =========================================================
def process_feature_map(tensor, target_size):
    if tensor.shape[1] > 1:
        tensor = torch.mean(tensor, dim=1, keepdim=True)
    feature = tensor[0, 0].numpy()

    # 极值归一化 (0到1之间)
    feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)


    feature = (feature * 255).astype(np.uint8)
    feature_resized = cv2.resize(feature, target_size)
    heatmap = cv2.applyColorMap(feature_resized, cv2.COLORMAP_JET)
    return heatmap


def visualize_and_save(original_bgr, save_path="dsscm_surgery_vis.jpg"):
    h, w = original_bgr.shape[:2]

    # 提取并处理刚才拦截到的 4 张特征图
    img_raw_da = process_feature_map(DSSCM_FEATURES['s_da_prime'], (w, h))
    img_knife = process_feature_map(DSSCM_FEATURES['det_presence_map'], (w, h))
    img_mold = process_feature_map(DSSCM_FEATURES['ll_guide_map'], (w, h), invert=True)
    img_final = process_feature_map(DSSCM_FEATURES['s_da_modulated_final'], (w, h))

    # 将热力图与原图半透明叠合 (0.5 原图 + 0.5 热力图)
    alpha = 0.5
    vis_1 = cv2.addWeighted(original_bgr, alpha, img_raw_da, 1 - alpha, 0)
    vis_2 = cv2.addWeighted(original_bgr, alpha, img_knife, 1 - alpha, 0)
    vis_3 = cv2.addWeighted(original_bgr, alpha, img_mold, 1 - alpha, 0)
    vis_4 = cv2.addWeighted(original_bgr, alpha, img_final, 1 - alpha, 0)

    # 水平拼接并加上边框分隔 
    concat_img = np.hstack([original_bgr, vis_1, vis_2, vis_3, vis_4])

    cv2.imwrite(save_path, concat_img)
    print(f"\nDSSCM 空间图已保存至: {save_path}")


# =========================================================
# 🚀 4. 主干逻辑：数据加载与推理 
# =========================================================
if __name__ == "__main__":
 
    weights_path = '/home/zeb/PythonProjects/FGC-Net/runs/Bdd100k-fg/_2026-03-12-23-00/epoch-240.pth' # 
    image_path = '/home/zeb/data/bdd100kfg/images/val/b2d502aa-64d3e228.jpg' 
    filename = image_path.split('/')[-1]
    save_name = f"dsscm_{filename}.jpg"
    img_size = 640
    # ===================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 锁死 CPU


    # 1. 初始化模型并加载权重
    class MockCfg:
        pass  # 弄个假的 cfg 防止 get_net 报错


    model = get_net(MockCfg())

    print(f"📦 正在加载权重: {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # 2. 自动搜寻 DSSCM 模块并挂上窃听器
    target_module = None
    for m in model.model:
        if m.__class__.__name__ == 'DetSegSpatialConsistencyModule':
            target_module = m
            break

    if target_module is None:
        raise ValueError("❌ 未在 model.model 里找到 DSSCM 模块！请检查你的模型结构。")

    old_fw = hook_dsscm_forward(target_module)


    # 3. 图像预处理
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"找不到图片: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))  # resize 到 640

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img_resized).unsqueeze(0).to(device)  # [1, 3, 640, 640]

    # 4. 执行前向传播
    print("进行前向推演...")
    with torch.no_grad():
        _ = model(img_tensor)

    # 5. 生成可视化大图
    visualize_and_save(img_bgr, save_name)


    target_module.forward = old_fw
    print("结束！")