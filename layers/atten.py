import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat

try:
    from mamba_ssm import Mamba
except ImportError:
    pass


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Standard Transformer multi-head self-attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Linear projections and split into multiple heads
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Rearrange to (batch, heads, length, dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # Broadcast mask along batch and head dimensions
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Combine heads back to (batch, length, model_dim)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class FeedForward(nn.Module):
    """Standard Transformer feed-forward layer with Pre-LN and residual connection"""

    def __init__(self, d_model, d_hidden, dropout=0.1, activation="relu"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        norm_x = self.norm(x)
        output = self.fc2(self.activation(self.fc1(norm_x)))
        return x + self.dropout(output)


class EncoderLayer_v2(nn.Module):
    """
    Modified Transformer Encoder Layer with Temporal Attention (optimized)
    Uses Pre-LN structure:
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(
            self,
            d_time,
            d_model,
            d_inner,
            n_head,
            d_k,
            d_v,
            diagonal_attention_mask,
            device,
            dropout=0.1,
            attn_dropout=0.1,
            activation="relu",
    ):
        super().__init__()

        self.diagonal_attention_mask = diagonal_attention_mask
        self.device = device
        self.d_time = d_time

        # ==== 1) Standard Multi-Head Self-Attention ====
        self.norm1 = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout1 = nn.Dropout(dropout)

        # ==== 2) Temporal Attention ====
        self.norm2 = nn.LayerNorm(d_model)
        self.t_attn = TemporalAttention(d_model, d_k, d_model)
        self.dropout2 = nn.Dropout(dropout)

        # ==== 3) Two Feed-Forward Networks ====
        self.ffn1 = FeedForward(d_model, d_inner, dropout, activation)
        self.ffn2 = FeedForward(d_model, d_inner, dropout, activation)

    def forward(self, x, time_points):
        mask_time = torch.eye(self.d_time, device=self.device,
                              dtype=torch.bool) if self.diagonal_attention_mask else None

        # # Pre-LN -> Multi-Head Attention -> Add -> FFN
        # norm_x = self.norm1(x)
        # attn_out, attn_weights_self = self.slf_attn(norm_x, norm_x, norm_x, attn_mask=mask_time)
        # x = x + self.dropout1(attn_out)
        # x = self.ffn1(x)

        # Pre-LN -> Temporal Attention -> Add -> FFN
        norm_x = self.norm2(x)
        t_out, attn_weights_temporal = self.t_attn(norm_x, time_points)
        x = x + self.dropout2(t_out)
        x = self.ffn2(x)

        return x, attn_weights_temporal


class MambaEncoderLayer(nn.Module):
    """Encoder layer using the Mamba module instead of attention"""

    def __init__(
        self,
        d_time,
        d_model,
        d_inner,
        device,
        dropout=0.1,
        activation="relu",
        use_norm=True,
    ):
        super().__init__()
        self.d_time = d_time
        self.device = device
        self.use_norm = use_norm

        # ==== 1) Mamba Module ====
        # Mamba replaces attention with global dependency modeling
        self.norm1 = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=4,       # state dimension, tunable
            d_conv=4,        # local convolution channels for short-term dependencies
            expand=2,        # channel expansion ratio
        )
        self.dropout1 = nn.Dropout(dropout)

        # ==== 2) Feed-Forward Network ====
        self.norm2 = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # 1) Mamba layer replaces self-attention
        residual = x
        x = self.norm1(x)
        x = self.mamba(x)  # [B, T, D]
        x = residual + self.dropout1(x)

        # 2) Feed-forward layer
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout2(x)

        return x, None  # keep same interface as EncoderLayer


class PositionWiseFeedForward(nn.Module):
    """Standard position-wise feed-forward network"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    """Standard Transformer encoder layer (Pre-LN version)"""

    def __init__(
            self,
            d_time,
            d_feature,
            d_model,
            d_inner,
            n_head,
            d_k,
            d_v,
            diagonal_attention_mask,
            device,
            dropout=0.1,
            attn_dropout=0.1,
    ):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = diagonal_attention_mask
        self.device = device
        self.d_time = d_time
        self.d_feature = d_feature
        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(self.device)
        else:
            mask_time = None

        residual = enc_input
        # Pre-LN before attention computation
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=mask_time
        )
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)

        return enc_output, attn_weights


class TemporalAttention(nn.Module):
    """Temporal attention module that adds time-based bias"""

    def __init__(self, d_model, d_k, time_emb_dim):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_k)
        self.W_t = nn.Linear(time_emb_dim, d_k)
        self.W_o = nn.Linear(d_k, d_model)

    def forward(self, x, time_points):
        B, L, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        T = self.W_t(time_points)  # (B, L, d_k)

        # Standard temporal attention: add time bias to QK^T
        scores = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        time_bias = torch.matmul(Q, T.transpose(1, 2))  # (B, L, L)
        scores = scores + time_bias

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return self.W_o(output), attn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                    for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, dim: int = 1, return_only_pos: bool = False):
        pos_enc = self.pos_table[:, :x.size(dim)].clone().detach()
        if return_only_pos:
            return pos_enc
        x_with_pos = x + pos_enc
        return x_with_pos


class AttentionLayer(nn.Module):
    """Multi-head scaled dot-product attention"""

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Split into heads
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim ** 0.5

        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    """Canonical self-attention layer"""

    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class EmbeddedAttention(nn.Module):
    """Spatial embedded attention layer"""

    def __init__(self, model_dim, adaptive_embedding_dim):
        super().__init__()

        self.model_dim = model_dim
        self.FC_Q_K = nn.Linear(adaptive_embedding_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, value, emb):
        # value: (batch_size, ..., seq_length, model_dim)
        # emb:   (..., length, model_dim)
        batch_size = value.shape[0]
        query = self.FC_Q_K(emb)
        key = self.FC_Q_K(emb)
        value = self.FC_V(value)

        key = key.transpose(-1, -2)

        # Re-normalization
        query = torch.softmax(query, dim=-1)
        key = torch.softmax(key, dim=-1)
        query = repeat(query, 'n s1 s2 -> b n s1 s2', b=batch_size)
        key = repeat(key, 'n s2 s1 -> b n s2 s1', b=batch_size)

        out = key @ value
        out = query @ out
        return out
