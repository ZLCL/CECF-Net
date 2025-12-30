import torch.nn as nn
import torch
import torch.nn.functional as F

from layers.atten import PositionalEncoding


class MultiScaleStochasticResampler(nn.Module):
    """
    Vectorized Bayesian Resampling Module (no for-loops over batch)
    """

    def __init__(self, d_model, n_scales=2, min_scale=0.5, max_scale=2.0,
                 max_sigma=0.5, mode="linear"):
        super().__init__()
        self.n_scales = n_scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_sigma = max_sigma
        self.mode = mode

        hidden = max(4, d_model // 2)

        self.mu_net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_scales),
            nn.Sigmoid()
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_scales),
            nn.Softplus()
        )

    def forward(self, X):
        """
        X: (B, T, D)
        return: list of (B, T, D)
        """
        device = X.device
        B, T, D = X.shape

        # ---- 1) μ and σ prediction from global context ----
        context = X.mean(dim=1)  # (B, D)

        mu_raw = self.mu_net(context)                  # (B, n_scales)
        sigma_raw = self.sigma_net(context) + 1e-6     # (B, n_scales)

        # map μ ∈ [min_scale, max_scale]
        mu = self.min_scale + (self.max_scale - self.min_scale) * mu_raw

        # σ clip to avoid exploding sampling
        sigma = torch.clamp(sigma_raw, max=self.max_sigma)

        # ---- 2) sample scale factors s ~ N(μ, σ²) ----
        eps = torch.randn_like(mu)
        s_sample = mu + sigma * eps                    # (B, n_scales)
        s_sample = torch.clamp(s_sample, self.min_scale, self.max_scale)

        # ---- 3) Fully vectorized resampling ----
        # Permute to (B, D, T) for interpolate
        X_ch = X.permute(0, 2, 1)                      # (B, D, T)

        # Expand scale dimension
        # X_expanded: (B, n_scales, D, T)
        X_expanded = X_ch.unsqueeze(1).expand(B, self.n_scales, D, T)

        # Merge (B*n_scales) as batch dimension
        X_flat = X_expanded.reshape(B * self.n_scales, D, T)

        # Prepare target lengths
        # s_sample: (B, n_scales) → flatten → (B*n_scales,)
        s_flat = s_sample.reshape(B * self.n_scales)
        new_lens = torch.clamp((T * s_flat).round().long(), min=2)

        # For variable-length interpolate:
        # We loop only *per scale*, not per batch
        # But we still produce vectorized results, no B-loop

        X_resampled_list = []

        for i in range(self.n_scales):
            # indices for this scale
            idx = torch.arange(B, device=device) * self.n_scales + i
            X_sel = X_flat[idx]           # (B, D, T)
            lens_sel = new_lens[idx]      # (B,)

            # unique lengths → run interpolate only per unique new_len
            unique_lens = lens_sel.unique()

            X_back = torch.zeros(B, D, T, device=device)

            for L in unique_lens:
                mask = lens_sel == L
                if mask.sum() == 0:
                    continue

                X_chunk = X_sel[mask]  # (b', D, T)

                # INTERPOLATE TO L
                X_rs = F.interpolate(
                    X_chunk,
                    size=L.item(),
                    mode=self.mode,
                    align_corners=False if self.mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
                )
                # BACK TO T
                X_back_chunk = F.interpolate(
                    X_rs,
                    size=T,
                    mode=self.mode,
                    align_corners=False if self.mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
                )
                X_back[mask] = X_back_chunk

            # permute back to (B, T, D)
            X_resampled = X_back.permute(0, 2, 1)
            X_resampled_list.append(X_resampled)

        return X_resampled_list


class MultiScaleResampleFusion(nn.Module):

    def __init__(self, d_model, feature_num, n_scales):
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model

        # 输入特征映射到 d_model
        self.input_proj = nn.Linear(feature_num, d_model)

        # X_ori 残差映射
        self.residual_proj = nn.Linear(feature_num, d_model)

        # 多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True
        )

    def forward(self, X_ori, X_list):
        B, T, _ = X_list[0].shape

        # --- 多尺度输入投影 ---
        X_proj = [self.input_proj(x) for x in X_list]     # S 个 (B,T,d)
        X_stack = torch.stack(X_proj, dim=2)              # (B,T,S,d)
        X_in = X_stack.view(B * T, self.n_scales, self.d_model)

        # --- cross-scale attention ---
        out, attn_w = self.attn(
            X_in, X_in, X_in,
            need_weights=True,
            average_attn_weights=False
        )
        # 现在 attn_w: (B*T, h, S, S)

        BT, h, S, _ = attn_w.shape

        # --- 计算 alpha（scale 权重） ---
        alpha = attn_w.sum(dim=-1)          # (BT, h, S)
        alpha = alpha.mean(dim=1)           # (BT, S)
        alpha = torch.softmax(alpha, dim=-1)
        alpha = alpha.unsqueeze(-1)         # (BT, S, 1)

        # --- 融合多尺度 ---
        X_sum = (out * alpha).sum(dim=1)    # (BT, d)
        X_res = X_in.mean(dim=1)            # (BT, d)

        X_fused = X_sum + X_res             # (BT, d)
        X_fused = X_fused.view(B, T, self.d_model)

        # --- 主残差（原始序列） ---
        X_ori_proj = self.residual_proj(X_ori)  # (B,T,d)

        return X_fused + X_ori_proj
