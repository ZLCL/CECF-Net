import torch.nn as nn
import torch
import torch.nn.functional as F


def LinearTrunkGeneration(x, mask=None):
    """
    Perform vectorized linear interpolation using PyTorch.
    Modified:
        - head missing  -> mean
        - middle missing -> linear interpolation
        - tail missing  -> mean
    """
    B, T, D = x.shape
    device = x.device
    x = x.clone()

    # mask
    if mask is None:
        mask = ~torch.isnan(x)
    mask = mask.bool().to(device)

    # global mean per sample: (B,1,D)
    # if all missing, replace NaN -> 0 temporarily then compute mean
    x_tmp = torch.where(mask, x, torch.tensor(float("nan"), device=device))
    seq_mean = torch.nanmean(x_tmp, dim=1, keepdim=True)        # (B,1,D)
    seq_mean = torch.where(torch.isnan(seq_mean), torch.zeros_like(seq_mean), seq_mean)

    # x_interp buffer
    x_interp = x.clone()

    # ---------------- Forward fill LOV ----------------
    last_val = torch.zeros_like(x)
    last_idx = torch.zeros_like(x)

    for i in range(T):
        if i == 0:
            prev_val = seq_mean.squeeze(1)      # 如果第一个位置缺失，用均值
            prev_idx = torch.zeros(B, D, device=device)
        else:
            prev_val = last_val[:, i - 1, :]
            prev_idx = last_idx[:, i - 1, :]

        current_val = x[:, i, :]
        current_t = torch.full((B, D), i, device=device, dtype=torch.float32)

        last_val[:, i, :] = torch.where(mask[:, i, :], current_val, prev_val)
        last_idx[:, i, :] = torch.where(mask[:, i, :], current_t, prev_idx)

    lov = last_val.clone()

    # ---------------- Backward fill NOV ----------------
    next_val = torch.zeros_like(x)
    next_idx = torch.zeros_like(x)

    for i in reversed(range(T)):
        if i == T - 1:
            next_val_curr = seq_mean.squeeze(1)     # 尾端缺失 -> 均值
            next_idx_curr = torch.full((B, D), T - 1, device=device, dtype=torch.float32)
        else:
            next_val_curr = next_val[:, i + 1, :]
            next_idx_curr = next_idx[:, i + 1, :]

        current_val = x[:, i, :]
        current_t = torch.full((B, D), i, device=device, dtype=torch.float32)

        next_val[:, i, :] = torch.where(mask[:, i, :], current_val, next_val_curr)
        next_idx[:, i, :] = torch.where(mask[:, i, :], current_t, next_idx_curr)

    nov = next_val.clone()

    # ---------------- Linear interpolation for middle ----------------
    missing = ~mask
    t_range = torch.arange(T, device=device).view(1, T, 1).expand(B, T, D)

    weight = (t_range - last_idx) / (next_idx - last_idx + 1e-6)
    lin_x = last_val + weight * (next_val - last_val)

    # only fill middle-missing by linear interpolation
    x_interp[missing] = lin_x[missing]

    # ---------------- Fill head / tail continuous missing by mean ----------------
    # head
    for b in range(B):
        for d in range(D):
            obs_pos = torch.where(mask[b, :, d])[0]
            # 如果全缺失，则全部填均值
            if len(obs_pos) == 0:
                x_interp[b, :, d] = seq_mean[b, 0, d]
                continue

            first_obs = obs_pos[0]
            # head
            if first_obs > 0:
                x_interp[b, :first_obs, d] = seq_mean[b, 0, d]

            # tail
            last_obs = obs_pos[-1]
            if last_obs < T - 1:
                x_interp[b, last_obs + 1:, d] = seq_mean[b, 0, d]

    # Return final
    return x_interp, lov, nov


class MissingInfoEnhenced(nn.Module):
    def __init__(self, d_in, num_freq, max_time, neighbor_size=15):
        super().__init__()
        self.d_in = d_in
        self.neighbor_size = neighbor_size
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.mask_embedding = MaskEmbedding(num_freq=num_freq, max_time=max_time)

        # 输入维度：X + lov + nov + mask_embed + max_context + min_context
        # X/lov/nov: D
        # mask_emb: D*(1+2*num_freq)
        # max_context_boundary: D
        # min_context_boundary: D
        concat_dim = d_in * 3 + d_in * (1 + 2 * num_freq)
        concat_dim_upd = d_in + d_in * (1 + 2 * num_freq)

        # 竞争门控（lov / nov）
        self.comp_gate = nn.Sequential(
            nn.Linear(concat_dim, d_in),
            nn.ReLU(),
            nn.Linear(d_in, 2 * d_in)
        )

        # 更新门控
        self.update_gate = nn.Sequential(
            nn.Linear(concat_dim_upd, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_in),
            nn.Sigmoid()
        )

        # 候选上下文
        self.candidate_net = nn.Sequential(
            nn.Linear(concat_dim, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_in)
        )

    def forward(self, X, lov, nov, masks):
        B, T, D = X.shape
        miss_flag = 1 - masks

        # ============================================================
        # 1. 计算缺失片段首尾
        # ============================================================
        miss_flag_int = miss_flag.int()
        miss_start = (miss_flag_int * (1 - F.pad(miss_flag_int[:, :-1, :], (0, 0, 1, 0)))).bool()
        miss_end = (miss_flag_int * (1 - F.pad(miss_flag_int[:, 1:, :], (0, 0, 0, 1)))).bool()
        boundary_mask = (miss_start | miss_end).float()

        # ============================================================
        # 2. 邻域 max/min 计算
        # ============================================================
        N = self.neighbor_size
        max_context = torch.zeros_like(X)
        min_context = torch.zeros_like(X)

        for t in range(T):
            left = max(0, t-N)
            right = min(T, t+N+1)

            neigh_vals = X[:, left:right, :] * masks[:, left:right, :]
            neigh_mask = masks[:, left:right, :]

            valid = neigh_mask.sum(dim=1, keepdim=True) > 0

            max_v = torch.where(
                valid, neigh_vals.max(dim=1, keepdim=True).values, X[:, t:t+1, :]
            )
            min_v = torch.where(
                valid, neigh_vals.min(dim=1, keepdim=True).values, X[:, t:t+1, :]
            )

            max_context[:, t:t+1, :] = max_v
            min_context[:, t:t+1, :] = min_v

        # ============================================================
        # 3. 仅在缺失边界注入 max/min 先验
        # ============================================================
        max_context_boundary = max_context * boundary_mask
        min_context_boundary = min_context * boundary_mask

        # ============================================================
        # 4. 生成 mask embedding
        # ============================================================
        mask_emb = self.mask_embedding(masks)
        m_emb_flat = mask_emb.view(B, T, D * 11)

        # ============================================================
        # 5. 拼接所有输入（加入 max/min）
        # ============================================================
        inp_comp = torch.cat([
            X,                 # D
            lov,               # D
            nov,               # D
            m_emb_flat,  # D*(1+2freq)
        ], dim=-1)

        # ============================================================
        # 6. 竞争门控
        # ============================================================
        comp_logits = self.comp_gate(inp_comp).view(B, T, 2, D)
        comp_weight = torch.softmax(comp_logits, dim=2)
        gate_lov, gate_nov = comp_weight[:, :, 0, :], comp_weight[:, :, 1, :]

        context_raw = gate_lov * (lov - X) + gate_nov * (nov - X)

        inp_candidate = torch.cat([
            X,                 # D
            max_context_boundary,  # D
            min_context_boundary,  # D
            m_emb_flat,  # D*(1+2freq)
        ], dim=-1)
        # 候选上下文 + max/min 先验
        candidate_context = self.candidate_net(inp_candidate) + context_raw

        # ============================================================
        # 7. 更新门控
        # ============================================================
        inp_upd = torch.cat([
            X,                 # D
            m_emb_flat,  # D*(1+2freq)
        ], dim=-1)
        z = self.update_gate(inp_upd)
        X_new = (1 - z) * X + z * candidate_context

        # ============================================================
        # 8. 仅增强缺失位置
        # ============================================================
        X_enhenced = X + self.alpha * miss_flag * (X_new - X)
        return X_enhenced



class MaskEmbedding(nn.Module):
    def __init__(self, num_freq: int = 10, max_time: int = 500, include_input: bool = True):
        """
        Time Series Positional / Mask Encoding using multi-frequency sin/cos.

        Args:
            num_freq: Number of frequency bands.
            max_time: Maximum time value for normalization.
            include_input: Whether to include the original input in the encoding.
        """
        super().__init__()
        self.num_freq = num_freq
        self.include_input = include_input
        self.max_time = max_time

        # Frequency bands: exponential growth, can also be linear if preferred
        self.freq_bands = 2.0 ** torch.linspace(0, num_freq - 1, num_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, D), can be time indices or mask values.

        Returns:
            Encoded tensor (B, T, D, encoding_dim), where encoding_dim = 1 + 2*num_freq if include_input=True
        """
        B, T, D = x.shape

        # Normalize time/mask values to [0,1]
        x_norm = x / max(self.max_time - 1, 1)
        x_expanded = x_norm.unsqueeze(-1)  # (B, T, D, 1)

        # Initialize encoding list
        encodings = [x_expanded] if self.include_input else []

        # Apply sin/cos for each frequency
        for freq in self.freq_bands.to(x.device):
            encodings.append(torch.sin(freq * x_expanded))
            encodings.append(torch.cos(freq * x_expanded))

        # Concatenate along the last dimension
        return torch.cat(encodings, dim=-1)  # (B, T, D, encoding_dim)
