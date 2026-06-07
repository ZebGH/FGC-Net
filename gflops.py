
import torch
from thop import profile
from thop import clever_format
from lib.models.v10v8 import get_net

# 1. 初始化模型
# 根据你提供的代码，这里实例化 YOLOP 模型
model = get_net(cfg=False)
model.eval()  # 切换到推理模式

# 2. 构造标准输入张量 (Batch Size=1, Channels=3, H=640, W=640)
# 根据你的设定，推理尺寸对齐为 640x640
dummy_input = torch.randn(1, 3, 640, 640)

# 3. 使用 thop 计算 Params 和 MACs
print("正在计算 Params 和 GFLOPs，请稍候...")
macs, params = profile(model, inputs=(dummy_input,), verbose=False)

# 4. 格式化输出 (1 MAC 约等于 2 FLOPs)
gflops = (macs * 2) / 1e9
params_m = params / 1e6

print("-" * 50)
print(f"Model: v10v8")
print(f"Input Resolution: 640x640")
print(f"Params (M): {params_m:.2f} M")
print(f"GFLOPs: {gflops:.2f}")
print("-" * 50)