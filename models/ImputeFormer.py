import torch.nn as nn
import torch

from layers.Normalization import nonstationary_norm, nonstationary_denorm
from utils.utils import masked_mae_cal
from layers.mlp import MLP
from einops import repeat
from layers.atten import AttentionLayer, EmbeddedAttention


class EmbeddedAttentionLayer(nn.Module):
    """
    Spatial embedded attention layer
    """

    def __init__(self,
                 model_dim, adaptive_embedding_dim, feed_forward_dim=2048, dropout=0):
        super().__init__()

        self.attn = EmbeddedAttention(model_dim, adaptive_embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim))

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, emb, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        # emb: (..., length, model_dim)
        residual = x
        out = self.attn(x, emb)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class ProjectedAttentionLayer(nn.Module):
    """
    Temporal projected attention layer
    """

    def __init__(self, seq_len, dim_proj, d_model, n_heads, d_ff=None, dropout=0.1):
        super(ProjectedAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.projector = nn.Parameter(torch.randn(dim_proj, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

        self.seq_len = seq_len

    def forward(self, x):
        # x: [b s n d]
        batch = x.shape[0]
        projector = repeat(self.projector, 'dim_proj d_model -> repeat seq_len dim_proj d_model',
                           repeat=batch, seq_len=self.seq_len)  # [b, s, c, d]

        message_out = self.out_attn(projector, x, x)  # [b, s, c, d] <-> [b s n d] -> [b s c d]
        message_in = self.in_attn(x, projector, message_out)  # [b s n d] <-> [b, s, c, d] -> [b s n d]
        message = x + self.dropout(message_in)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message


class ImputeFormerModel(nn.Module):
    """
    Spatiotempoarl Imputation Transformer
    """

    def __init__(
            self,
            configs,
    ):
        super(ImputeFormerModel, self).__init__()
        self.apply_nonstationary_norm = configs.apply_nonstationary_norm
        self.MIT = configs.MIT
        self.num_nodes = configs.num_nodes
        self.seq_len = configs.seq_len
        self.input_dim = configs.input_dim
        self.output_dim = configs.output_dim
        self.input_embedding_dim = configs.input_embedding_dim
        self.learnable_embedding_dim = configs.learnable_embedding_dim
        self.model_dim = (
                self.input_embedding_dim
                + self.learnable_embedding_dim)
        self.num_temporal_heads = configs.num_temporal_heads
        self.num_layers = configs.num_layers

        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        self.dim_proj = configs.dim_proj

        self.learnable_embedding = nn.init.xavier_uniform_(
            nn.Parameter(
                torch.empty(self.seq_len, self.num_nodes, self.learnable_embedding_dim)))

        self.readout = MLP(self.model_dim, self.model_dim, self.output_dim, n_layers=2)

        self.attn_layers_t = nn.ModuleList(
            [ProjectedAttentionLayer(self.num_nodes, self.dim_proj, self.model_dim, configs.num_temporal_heads,
                                     self.model_dim, configs.dropout)
             for _ in range(self.num_layers)])

        self.attn_layers_s = nn.ModuleList(
            [EmbeddedAttentionLayer(self.model_dim, self.learnable_embedding_dim,
                                    configs.feed_forward_dim)
             for _ in range(self.num_layers)])

    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, masks)
        batch_size = X.shape[0]
        X = X.unsqueeze(-1)  # [b s n c=1]
        masks = masks.unsqueeze(-1)  # [b s n c=1]

        # Whiten missing values
        X = X * masks
        X = self.input_proj(X)  # (batch_size, in_steps, num_nodes, input_embedding_dim)

        node_emb = self.learnable_embedding.expand(batch_size, *self.learnable_embedding.shape)
        X = torch.cat([X, node_emb], dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        X = X.permute(0, 2, 1, 3)  # [b n s c]
        for att_t, att_s in zip(self.attn_layers_t, self.attn_layers_s):
            X = att_t(X)
            X = att_s(X, self.learnable_embedding, dim=1)

        X = X.permute(0, 2, 1, 3)  # [b s n c]
        x_tilde = self.readout(X)

        x_tilde = x_tilde.squeeze(-1)  # [b s n]
        masks = masks.squeeze(-1)  # [b s n]

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            x_tilde = nonstationary_denorm(x_tilde, means, stdev)
        # Below is the SAITS processing pipeline:
        # replace the observed part with values from X
        X_c = masks * inputs["X"] + (1 - masks) * x_tilde
        return X_c, x_tilde

    def forward(self, inputs, stage):
        X, masks = inputs['X'], inputs['missing_mask']

        imputed_data, reconstruction = self.impute(inputs)
        reconstruction_MAE = masked_mae_cal(reconstruction, X, masks)

        if (self.MIT or stage == "val") and stage != "test":
            imputation_MAE = masked_mae_cal(
                reconstruction, inputs.get("X_holdout"), inputs.get("indicating_mask")
            )
        else:
            imputation_MAE = torch.tensor(0.0, device=X.device)

        reconstruction_loss = reconstruction_MAE
        imputation_loss = imputation_MAE
        # ensemble the results as a dictionary for return
        results = {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_loss,
            "imputation_loss": imputation_loss,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }

        return results
