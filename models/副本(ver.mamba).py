from layers.atten import *
from utils.utils import masked_mae_cal


def linear_interpolate_tensor(x, mask=None):
    """
    纯 PyTorch 向量化线性插值 (B,T,D)
    完整展开时间维度，避免索引报错
    """
    B, T, D = x.shape
    device = x.device
    x = x.clone()

    if mask is None:
        mask = ~torch.isnan(x)
    mask = mask.bool().to(device)

    x_interp = x.clone()

    # 前向填充
    last_val = torch.zeros_like(x, device=device)
    last_idx = torch.zeros_like(x, device=device)
    for i in range(T):
        if i == 0:
            prev_val = torch.zeros(B, D, device=device)
            prev_idx = torch.zeros(B, D, device=device)
        else:
            prev_val = last_val[:, i - 1, :]
            prev_idx = last_idx[:, i - 1, :]

        current_val = x[:, i, :]
        current_t = torch.full((B, D), i, device=device, dtype=torch.float32)

        last_val[:, i, :] = torch.where(mask[:, i, :], current_val, prev_val)
        last_idx[:, i, :] = torch.where(mask[:, i, :], current_t, prev_idx)

    # 后向填充
    next_val = torch.zeros_like(x, device=device)
    next_idx = torch.zeros_like(x, device=device)
    for i in reversed(range(T)):
        if i == T - 1:
            next_val_curr = torch.zeros(B, D, device=device)
            next_idx_curr = torch.full((B, D), T - 1, device=device, dtype=torch.float32)
        else:
            next_val_curr = next_val[:, i + 1, :]
            next_idx_curr = next_idx[:, i + 1, :]

        current_val = x[:, i, :]
        current_t = torch.full((B, D), i, device=device, dtype=torch.float32)

        next_val[:, i, :] = torch.where(mask[:, i, :], current_val, next_val_curr)
        next_idx[:, i, :] = torch.where(mask[:, i, :], current_t, next_idx_curr)

    # 线性插值
    missing = ~mask
    weight = (torch.arange(T, device=device).view(1, T, 1).expand(B, T, D) - last_idx) / (
            next_idx - last_idx + 1e-6)
    x_interp[missing] = last_val[missing] + weight[missing] * (next_val[missing] - last_val[missing])

    return x_interp


def compute_last_observed_value(X, masks):
    """
    X: (B,T,D), masks: (B,T,D) 1=observed, 0=missing
    Returns lov: (B,T,D) where lov[b,t,d] is the last observed value at index < t,
    and 0 if none.
    """
    B, T, D = X.shape
    device = X.device

    idx_arange = torch.arange(T, device=device).view(1, T, 1)  # (1,T,1)
    pos = masks.long() * (idx_arange + 1)  # 1-based positions, 0 means no obs

    cummax_vals, _ = pos.cummax(dim=1)  # last seen position up to current (1-based or 0)
    # shift right to get last-before-current
    last_idx_prev = torch.zeros_like(cummax_vals)
    if T > 1:
        last_idx_prev[:, 1:, :] = cummax_vals[:, :-1, :]

    # convert to 0-based indices for gather; positions with 0 mean "no prev"
    idx0 = (last_idx_prev - 1).clamp(min=0).long()  # (B,T,D)

    # prepare X for gather: (B, D, T)
    X_perm = X.permute(0, 2, 1)
    idx0_perm = idx0.permute(0, 2, 1)

    gathered = torch.gather(X_perm, dim=2, index=idx0_perm)  # (B, D, T)
    lov = gathered.permute(0, 2, 1)  # (B,T,D)

    # zero out positions where there was no previous observation
    no_prev_mask = last_idx_prev == 0
    lov = lov.masked_fill(no_prev_mask, 0.0)

    return lov


def compute_next_observed_value(X, masks):
    """
    向量化计算 NOV（next observed value after current time step）。
    方法：把时间轴翻转，复用 compute_last_observed_value，然后再翻回。
    NOV[:, t, :] 是 t 时刻之后（>= t+1）第一个被观测到的值（若无则 0）。
    """
    # 翻转时间轴
    X_rev = X.flip(dims=[1])
    masks_rev = masks.flip(dims=[1])

    # 在翻转序列上计算 "last observed value before current in reversed" -> 对应原序列的 next
    lov_rev = compute_last_observed_value(X_rev, masks_rev)

    # 翻回来
    nov = lov_rev.flip(dims=[1])
    return nov


class MaskEmbedding(nn.Module):
    def __init__(self, feature_dim, emb_dim):
        super().__init__()
        # feature_dim is number of channels per feature to embed (1)
        self.linear = nn.Linear(1, emb_dim)
        self.layernorm = nn.LayerNorm(emb_dim)

    def forward(self, M):
        # M: B,T,D
        B, T, D = M.shape
        m = M.unsqueeze(-1)
        out = self.linear(m)
        out = self.layernorm(out)
        return out  # B,T,D,emb_dim


class IFSAmodel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_groups = configs.n_groups
        self.input_with_mask = configs.input_with_mask
        self.feature_num = configs.feature_num
        self.MIT = configs.MIT
        self.interpolate_with_mask = configs.interpolate_with_mask
        self.device = configs.device
        self.diagonal_attention_mask = configs.diagonal_attention_mask
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_inner = configs.d_inner
        self.n_head = configs.n_head
        self.d_k = configs.d_k
        self.d_v = configs.d_v
        self.dropoutr = configs.dropout
        self.dropout = nn.Dropout(p=self.dropoutr)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.mask_emb_dim = getattr(configs, "mask_emb_dim", 16)
        self.fused_dim = self.feature_num * 3 + self.feature_num * self.mask_emb_dim
        self.position_enc = PositionalEncoding(self.d_model, n_position=self.seq_len)
        self.mask_embedding = MaskEmbedding(1, self.mask_emb_dim)
        self.fuse_linear = nn.Linear(self.fused_dim, self.d_model)
        self.reduce_dim_z = nn.Linear(self.d_model, self.feature_num)
        self.low_pass_n = configs.low_pass_n
        self.encoder = MambaEncoderLayer(
            d_time=self.seq_len,
            d_model=self.d_model,
            d_inner=self.d_inner,
            device=self.device,
        )

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]

        lov = compute_last_observed_value(X, masks)  # (B,T,D)
        nov = compute_next_observed_value(X, masks)  # (B,T,D)

        # embeddings
        m_emb = self.mask_embedding(masks)  # B,T,D,emb
        B, T, D = X.shape
        m_emb_flat = m_emb.view(B, T, D * self.mask_emb_dim)
        lov_flat = lov.view(B, T, D)
        nov_flat = nov.view(B, T, D)
        X_flat = X.view(B, T, D)
        fused = torch.cat([X_flat, lov_flat, nov_flat, m_emb_flat], dim=-1)  # (B,T,fused_dim)
        x_pre = self.fuse_linear(fused)  # -> (B,T,d_model)

        enc_output = self.position_enc(x_pre)
        time_points = self.layer_norm(enc_output)

        enc_output, _ = self.encoder(enc_output)

        x_tilde = self.reduce_dim_z(enc_output)
        X_c = X * masks + x_tilde * (1 - masks)

        return X_c, x_tilde

    def forward(self, inputs, stage):
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, x_tilde = self.impute(inputs)

        reconstruction_MAE = masked_mae_cal(x_tilde, X, masks)


        if (self.MIT or stage == "val") and stage != "test":
            imputation_MAE = masked_mae_cal(
                x_tilde, inputs.get("X_holdout"), inputs.get("indicating_mask")
            )
        else:
            imputation_MAE = torch.tensor(0.0, device=X.device)

        reconstruction_loss = reconstruction_MAE
        imputation_loss = imputation_MAE

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_loss,
            "imputation_loss": imputation_loss,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }
