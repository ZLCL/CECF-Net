import torch.nn as nn
import torch

from layers.Embed import DataEmbedding
from layers.Normalization import nonstationary_norm, nonstationary_denorm
from layers.TimesBlock import TimesBlock
from utils.utils import masked_mae_cal


class TimesNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.n_layers = configs.n_layers
        self.dropout = configs.dropout
        self.d_model = configs.d_model
        self.n_features = configs.feature_num
        self.MIT = configs.MIT
        self.enc_embedding = DataEmbedding(
            self.n_features,
            self.d_model,
            dropout=self.dropout,
            n_max_steps=self.seq_len,
        )
        self.apply_nonstationary_norm = configs.apply_nonstationary_norm
        self.d_ffn = configs.d_ffn
        self.top_k = configs.top_k
        self.n_kernels = configs.n_kernels
        self.TimesBlock = nn.ModuleList(
            [TimesBlock(self.seq_len, 0, self.top_k, self.d_model, self.d_ffn, self.n_kernels)
             for _ in range(self.n_layers)]
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

        # for the imputation task, the output dim is the same as input dim
        self.projection = nn.Linear(self.d_model, self.n_features)

    def impute(self, inputs):
        X, missing_mask = inputs['X'], inputs['missing_mask']
        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)
        # embedding
        enc_out = self.enc_embedding(X)  # [B,T,C]
        # TimesNet processing
        for i in range(self.n_layers):
            enc_out = self.layer_norm(self.TimesBlock[i](enc_out))

        # project back the original data space
        reconstruction = self.projection(enc_out)
        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            reconstruction = nonstationary_denorm(reconstruction, means, stdev)

        imputed_data = missing_mask * inputs['X'] + (1 - missing_mask) * reconstruction

        return imputed_data, reconstruction

    def forward(self, inputs, stage):
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        imputed_data, reconstruction = self.impute(inputs)
        reconstruction_MAE = masked_mae_cal(reconstruction, X, missing_mask)

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
