import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.convnext import ConvNeXtBlock


class FocusedLinearAttention(nn.Module):
    """ 极致加速版的 O(N) 线性交叉注意力 (AMP 防爆版) """

    def __init__(self, query_dim, context_dim, output_dim):
        super().__init__()
        self.W_q = nn.Linear(query_dim, output_dim, bias=False)
        self.W_k = nn.Linear(context_dim, output_dim, bias=False)
        self.W_v = nn.Linear(context_dim, output_dim, bias=False)
        self.output_dim = output_dim

    def forward(self, query, context):
        with torch.cuda.amp.autocast(enabled=False):
            q_f32 = query.float()
            c_f32 = context.float()

            Q = F.relu(self.W_q(q_f32))
            K = F.relu(self.W_k(c_f32))
            V = self.W_v(c_f32)

            KV = torch.matmul(K.transpose(-2, -1), V)
            output = torch.matmul(Q, KV)

            norm_factor = torch.matmul(Q, K.sum(dim=1, keepdim=True).transpose(-2, -1)) + 1e-4
            output = output / norm_factor
        return output.to(query.dtype)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class HA_FIM(nn.Module):
    def __init__(self, det_channels, da_channels, ll_channels, proj_channels,
                 seg_enh_channels, det_enh_channels, out_channels_det, out_channels_da, out_channels_ll):
        super().__init__()
        self.proj_channels = proj_channels
        self.seg_enh_channels = seg_enh_channels
        self.det_enh_channels = det_enh_channels

        self.proj_det = nn.Conv2d(det_channels, proj_channels, 1)
        self.proj_da = nn.Conv2d(da_channels, proj_channels, 1)
        self.proj_ll = nn.Conv2d(ll_channels, proj_channels, 1)

        # 💡 核心修改：防淹没解耦，DA和LL拥有各自独立的增强卷积
        self.da_enh_conv = nn.Sequential(
            nn.Conv2d(proj_channels, seg_enh_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(seg_enh_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.ll_enh_conv = nn.Sequential(
            nn.Conv2d(proj_channels, seg_enh_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(seg_enh_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.det_enhancement = ConvNeXtBlock(in_chs=proj_channels)
        self.det_enh_proj = nn.Conv2d(proj_channels, det_enh_channels,
                                      1) if proj_channels != det_enh_channels else nn.Identity()

        self.cross_att_seg = FocusedLinearAttention(query_dim=seg_enh_channels, context_dim=det_enh_channels,
                                                    output_dim=seg_enh_channels)
        self.cross_att_seg_norm = nn.LayerNorm(seg_enh_channels)
        self.ffn_seg_final = MLP(seg_enh_channels, seg_enh_channels * 4, seg_enh_channels)
        self.norm_seg_final = nn.LayerNorm(seg_enh_channels)

        self.cross_att_det = FocusedLinearAttention(query_dim=det_enh_channels, context_dim=seg_enh_channels,
                                                    output_dim=det_enh_channels)
        self.cross_att_det_norm = nn.LayerNorm(det_enh_channels)
        self.ffn_det_final = MLP(det_enh_channels, det_enh_channels * 4, det_enh_channels)
        self.norm_det_final = nn.LayerNorm(det_enh_channels)

        self.dist_det = nn.Conv2d(det_enh_channels, out_channels_det, 1)
        self.dist_da = nn.Conv2d(seg_enh_channels, out_channels_da, 1)
        self.dist_ll = nn.Conv2d(seg_enh_channels, out_channels_ll, 1)

    def forward(self, x):
        feat_det, feat_da, feat_ll = x[0], x[1], x[2]
        B, _, H, W = feat_det.shape

        p_det = self.proj_det(feat_det)
        p_da = self.proj_da(feat_da)
        p_ll = self.proj_ll(feat_ll)

        # 💡 核心修改：独立增强后再拼接，保留高频细节
        feat_da_enh = self.da_enh_conv(p_da)
        feat_ll_enh = self.ll_enh_conv(p_ll)
        feat_seg_enhanced = torch.cat([feat_da_enh, feat_ll_enh], dim=1)

        feat_seg_enhanced_flat = feat_seg_enhanced.flatten(2).transpose(1, 2)

        feat_det_temp = self.det_enhancement(p_det)
        feat_det_enhanced = self.det_enh_proj(feat_det_temp)
        feat_det_enhanced_flat = feat_det_enhanced.flatten(2).transpose(1, 2)

        # Seg attends to Det
        feat_seg_cross = self.cross_att_seg(query=feat_seg_enhanced_flat, context=feat_det_enhanced_flat)
        feat_seg_interacted_flat = feat_seg_enhanced_flat + self.cross_att_seg_norm(feat_seg_cross)
        feat_seg_interacted_flat = feat_seg_interacted_flat + self.ffn_seg_final(
            self.norm_seg_final(feat_seg_interacted_flat))

        # Det attends to Seg
        feat_det_cross = self.cross_att_det(query=feat_det_enhanced_flat, context=feat_seg_enhanced_flat)
        feat_det_interacted_flat = feat_det_enhanced_flat + self.cross_att_det_norm(feat_det_cross)
        feat_det_interacted_flat = feat_det_interacted_flat + self.ffn_det_final(
            self.norm_det_final(feat_det_interacted_flat))

        feat_det_interacted = feat_det_interacted_flat.transpose(1, 2).reshape(B, self.det_enh_channels, H, W)
        feat_seg_interacted = feat_seg_interacted_flat.transpose(1, 2).reshape(B, self.seg_enh_channels, H, W)

        out_det = self.dist_det(feat_det_interacted)
        out_da = self.dist_da(feat_seg_interacted)
        out_ll = self.dist_ll(feat_seg_interacted)

        return out_det, out_da, out_ll