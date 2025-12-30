from layers.Normalization import nonstationary_norm, nonstationary_denorm
from layers.atten import *
from utils.utils import masked_mae_cal


def LinearTrunkGeneration(x, mask=None):
    """
    Vectorized linear interpolation that also returns LOV and NOV.

    Args:
        x: (B, T, D) input tensor
        mask: (B, T, D) binary mask (1=observed, 0=missing)

    Returns:
        x_interp: linearly interpolated tensor (B, T, D)
        lov: last observed value at each timestep (B, T, D)
        nov: next observed value at each timestep (B, T, D)
    """
    B, T, D = x.shape
    device = x.device
    x = x.clone()

    if mask is None:
        mask = ~torch.isnan(x)
    mask = mask.bool().to(device)

    x_interp = x.clone()

    # ===== Forward Fill (LOV) =====
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

    lov = last_val.clone()  # 保存 LOV

    # ===== Backward Fill (NOV) =====
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

    nov = next_val.clone()  # 保存 NOV

    # ===== Linear Interpolation =====
    missing = ~mask
    weight = (torch.arange(T, device=device).view(1, T, 1).expand(B, T, D) - last_idx) / (
            next_idx - last_idx + 1e-6)
    x_interp[missing] = last_val[missing] + weight[missing] * (next_val[missing] - last_val[missing])

    return x_interp, lov, nov


class BayesianResampleBlock(nn.Module):
    """
    Bayesian Resampling Module

    - Does NOT require or return any mask.
    - Dynamically predicts a set of (μ_i, σ_i) from the input sequence X.
    - Samples multiple scale factors s_i ~ N(μ_i, σ_i²).
    - For each scale:
         (1) Interpolates X to a new length new_len = s_i * T.
         (2) Re-interpolates the result back to the original length T.
    - Returns a list of resampled sequences at different scales.

    Returns:
        X_resampled_list: list of resampled tensors [(B, T, D), ...]
                          one per sampled scale.
        (kl_loss is removed in this version)
    """

    def __init__(self, d_model, n_scales=3, min_scale=0.5, max_scale=2.0, mode="linear"):
        super().__init__()
        self.n_scales = n_scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = mode

        hidden = max(4, d_model // 2)

        # Use the global mean of X as the contextual prior input
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
            nn.Softplus()  # ensures σ > 0
        )

    def forward(self, X):
        """
        Args:
            X: Tensor of shape (B, T, D)
               Input sequence.

        Returns:
            X_resampled_list: list of Tensors [(B, T, D), ...]
                Multi-scale resampled versions of X.
        """
        device = X.device
        B, T, D = X.shape

        # Compute global context from sequence mean
        context = X.mean(dim=1)  # (B, D)

        # Predict μ and σ for each scale
        mu_raw = self.mu_net(context)          # (B, n_scales)
        sigma_raw = self.sigma_net(context) + 1e-6  # avoid numerical zero

        # Map μ into [min_scale, max_scale] range
        mu = self.min_scale + (self.max_scale - self.min_scale) * mu_raw
        sigma = sigma_raw

        # Sample scale factors from N(μ, σ²)
        eps = torch.randn_like(mu)
        s_sample = mu + sigma * eps
        s_sample = torch.clamp(s_sample, self.min_scale, self.max_scale)  # (B, n_scales)

        X_resampled_list = []

        # Perform resampling for each scale
        for i in range(self.n_scales):
            X_back_list = []
            for b in range(B):
                s_b = s_sample[b, i].item()
                new_len = max(2, int(round(T * s_b)))

                # Interpolate along the temporal axis
                xb = X[b].permute(1, 0).unsqueeze(0)  # (1, D, T)
                xb_rs = F.interpolate(
                    xb, size=new_len, mode=self.mode,
                    align_corners=False if self.mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
                )
                # Re-interpolate back to original temporal length
                xb_back = F.interpolate(
                    xb_rs, size=T, mode=self.mode,
                    align_corners=False if self.mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
                )

                xb_back = xb_back.squeeze(0).permute(1, 0)  # (T, D)
                X_back_list.append(xb_back)

            X_back = torch.stack(X_back_list, dim=0).to(device)
            X_resampled_list.append(X_back)

        return X_resampled_list


class ResampleFusionBlock(nn.Module):
    """
    Multi-Scale Resample Fusion Block

    - Fuses multiple resampled versions of the same sequence.
    - Applies cross-scale attention to adaptively combine
      information from different temporal resampling levels.
    """

    def __init__(self, d_model, feature_num, n_scales):
        super().__init__()
        self.n_scales = n_scales
        self.feature_num = feature_num
        self.d_model = d_model

        # Project input feature dimension to model dimension
        self.input_proj = nn.Linear(feature_num, d_model)

        # Cross-scale attention is applied along the scale dimension
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

        # Fuse all scale representations into a single unified feature
        self.fc = nn.Linear(n_scales * d_model, d_model)

    def forward(self, X_list):
        """
        Args:
            X_list: list of Tensors [(B, T, D), ...]
                    Each element corresponds to one resampled scale.

        Returns:
            out: Tensor of shape (B, T, d_model)
                 Fused representation after cross-scale attention.
        """
        B, T, D = X_list[0].shape

        # Project each scale version into the same embedding space
        X_stack_proj = [self.input_proj(x) for x in X_list]  # (B, T, d_model)

        # Stack along the scale dimension
        X_stack = torch.stack(X_stack_proj, dim=2)  # (B, T, n_scales, d_model)

        # Apply cross-scale attention at each time step
        X_attn_in = X_stack.view(B * T, self.n_scales, self.d_model)  # (B*T, n_scales, d_model)
        X_attn_out, _ = self.attn(X_attn_in, X_attn_in, X_attn_in)    # (B*T, n_scales, d_model)

        # Concatenate all scales and fuse via linear projection
        X_attn_out = X_attn_out.reshape(B, T, self.n_scales * self.d_model)  # (B, T, n_scales*d_model)
        out = self.fc(X_attn_out)                                            # (B, T, d_model)

        return out


class MissingInfoEnhenced(nn.Module):
    def __init__(self, d_in, mask_emb_dim):
        super().__init__()
        # Gating networks for lov and nov enhancement
        self.lov_gate = nn.Sequential(
            nn.Linear(d_in * 3 + d_in * mask_emb_dim, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_in),
            nn.Sigmoid()
        )
        self.nov_gate = nn.Sequential(
            nn.Linear(d_in * 3 + d_in * mask_emb_dim, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_in),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.mask_emb_dim = mask_emb_dim
        self.mask_embedding = MaskEmbedding(self.mask_emb_dim)

    def forward(self, X, lov, nov, masks):
        """
        Args:
            X: Tensor of shape (B, T, D)
            lov, nov: Low- and high-variation reference signals, shape (B, T, D)
            masks: Missing indicator mask, shape (B, T, D)
        Returns:
            X_enhenced: Enhanced input sequence with missing-value awareness
        """
        B, T, D = X.shape
        miss_flag = (1 - masks)

        # Temporal difference along the time axis
        diff = X[:, 1:, :] - X[:, :-1, :]  # (B, T-1, D)
        # sign_change = diff[:, 1:, :] * diff[:, :-1, :]  # (B, T-2, D)

        # Detect local peaks and valleys
        # Peak: positive-to-negative transition
        # Valley: negative-to-positive transition
        peak_mask = (diff[:, :-1, :] > 0) & (diff[:, 1:, :] < 0)
        valley_mask = (diff[:, :-1, :] < 0) & (diff[:, 1:, :] > 0)

        # Align to input length by padding
        pad = (0, 0, 1, 1)  # pad 1 time step at both ends
        peak_mask = F.pad(peak_mask.float(), pad, mode='constant', value=0)  # (B, T, D)
        valley_mask = F.pad(valley_mask.float(), pad, mode='constant', value=0)  # (B, T, D)

        # Combine peak and valley features as local enhancement context
        peak_valley_feature = peak_mask * X + valley_mask * X  # (B, T, D)

        # Mask embedding for missing-value awareness
        mask_emb = self.mask_embedding(masks)
        m_emb_flat = mask_emb.view(B, T, D * self.mask_emb_dim)

        # Concatenate all inputs and compute adaptive gates
        inp = torch.cat([X, lov, nov, m_emb_flat], dim=-1)
        gate_lov = self.lov_gate(inp)
        gate_nov = self.nov_gate(inp)

        # Compute context fusion
        context = gate_lov * (lov - X) + gate_nov * (nov - X)

        # Apply enhancement only to missing positions
        X_enhenced = X + self.alpha * miss_flag * (context + peak_valley_feature)

        return X_enhenced


class MaskEmbedding(nn.Module):
    """
    Improved MaskEmbedding

    Args:
        emb_dim: int, embedding dimension per feature
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        # Map each mask value (0/1) to a vector of size emb_dim
        self.linear = nn.Linear(1, emb_dim)
        self.layernorm = nn.LayerNorm(emb_dim)
        self.activation = nn.ReLU()

    def forward(self, M):
        """
        Args:
            M: Tensor of shape (B, T, D), mask values 0/1

        Returns:
            out: Tensor of shape (B, T, D*emb_dim)
        """
        B, T, D = M.shape
        # Expand last dimension for linear mapping
        m = M.unsqueeze(-1)               # (B, T, D, 1)
        out = self.linear(m)              # (B, T, D, emb_dim)
        out = self.activation(out)        # Apply non-linearity
        out = self.layernorm(out)         # LayerNorm on embedding dimension
        # Flatten feature and embedding dimensions for concatenation
        out = out.reshape(B, T, D * self.emb_dim)  # (B, T, D*emb_dim)
        return out


class IFSAmodel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_groups = configs.n_groups
        self.input_with_mask = configs.input_with_mask
        self.feature_num = configs.feature_num
        self.MIT = configs.MIT
        self.device = configs.device
        self.diagonal_attention_mask = configs.diagonal_attention_mask
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_inner = configs.d_inner
        self.n_head = configs.n_head
        self.d_k = configs.d_k
        self.d_v = configs.d_v
        self.dropout_rate = configs.dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.mask_emb_dim = configs.mask_emb_dim
        self.fuse_linear = nn.Linear(self.d_model, self.d_model)
        self.fuse_linear_z = nn.Linear(self.feature_num * 3, self.d_model)
        self.read_out = nn.Linear(self.d_model, self.feature_num)
        self.low_pass_n = configs.low_pass_n
        self.apply_nonstationary_norm = configs.apply_nonstationary_norm

        self.position_enc = PositionalEncoding(self.d_model, n_position=self.seq_len)
        self.missing_enhencd = MissingInfoEnhenced(self.feature_num, self.mask_emb_dim)
        self.bayes_resampler = BayesianResampleBlock(
            d_model=self.feature_num,
            min_scale=0.5,
            max_scale=2.0,
            mode='linear'
        )
        self.resample_fusion = ResampleFusionBlock(d_model=self.d_model,
                                                   feature_num=self.feature_num, n_scales=3)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer_v2(
                    self.seq_len,
                    self.d_model,
                    self.d_inner,
                    self.n_head,
                    self.d_k,
                    self.d_v,
                    self.diagonal_attention_mask,
                    self.device,
                    self.dropout_rate,
                    0,
                )
                for _ in range(self.n_groups)
            ]
        )

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            X, means, stdev = nonstationary_norm(X, masks)

        trunk, lov, nov = LinearTrunkGeneration(X, masks)

        X_stem = self.missing_enhencd(trunk, lov, nov, masks)

        X_resampled_list = self.bayes_resampler(X_stem)

        X_fused = self.resample_fusion(X_resampled_list)

        enc_output = self.position_enc(X_fused)
        time_points = self.layer_norm(enc_output)
        for encoder_layer in self.layer_stack:
            enc_output, _ = encoder_layer(enc_output, time_points)
        reconstruction = self.read_out(enc_output)

        if self.apply_nonstationary_norm:
            reconstruction = nonstationary_denorm(reconstruction, means, stdev)

        X_c = X * masks + reconstruction * (1 - masks)

        return X_c, reconstruction

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
