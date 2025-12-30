from layers.Normalization import nonstationary_denorm, nonstationary_norm
from layers.atten import *
from utils.utils import masked_mae_cal


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        configs
    ):
        super().__init__()
        self.apply_nonstationary_norm = configs.apply_nonstationary_norm
        self.n_groups = configs.n_groups
        self.n_group_inner_layers = configs.n_group_inner_layers
        self.input_with_mask = configs.input_with_mask
        self.feature_num = configs.feature_num
        self.actual_feature_num = self.feature_num * 2 if self.input_with_mask else self.feature_num
        self.param_sharing_strategy = configs.param_sharing_strategy
        self.MIT = configs.MIT
        self.seq_len = configs.seq_len
        self.device = configs.device
        self.d_model = configs.d_model
        self.d_inner = configs.d_inner
        self.n_head = configs.n_head
        self.d_k = configs.d_k
        self.d_v = configs.d_v
        self.diagonal_attention_mask = configs.diagonal_attention_mask
        self.dropoutr = configs.dropout

        if configs.param_sharing_strategy == "between_group":
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack = nn.ModuleList(
                [
                    EncoderLayer(
                        self.seq_len,
                        self.actual_feature_num,
                        self.d_model,
                        self.d_inner,
                        self.n_head,
                        self.d_k,
                        self.d_v,
                        self.diagonal_attention_mask,
                        self.device,
                        self.dropoutr,
                        self.dropoutr,
                    )
                    for _ in range(self.n_group_inner_layers)
                ]
            )
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack = nn.ModuleList(
                [
                    EncoderLayer(
                        self.seq_len,
                        self.actual_feature_num,
                        self.d_model,
                        self.d_inner,
                        self.n_head,
                        self.d_k,
                        self.d_v,
                        self.diagonal_attention_mask,
                        self.device,
                        self.dropoutr,
                        self.dropoutr,
                    )
                    for _ in range(self.n_groups)
                ]
            )

        self.embedding = nn.Linear(self.actual_feature_num, self.d_model)
        self.position_enc = PositionalEncoding(self.d_model, n_position=self.seq_len)
        self.dropout = nn.Dropout(p=self.dropoutr)
        self.reduce_dim = nn.Linear(self.d_model, self.feature_num)

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]
        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, masks)
        input_X = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        learned_presentation = self.reduce_dim(enc_output)
        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            learned_presentation = nonstationary_denorm(learned_presentation, means, stdev)
        imputed_data = (
            masks * X + (1 - masks) * learned_presentation
        )  # replace non-missing part with original data
        return imputed_data, learned_presentation

    def forward(self, inputs, stage):
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, learned_presentation = self.impute(inputs)
        reconstruction_MAE = masked_mae_cal(learned_presentation, X, masks)
        if (self.MIT or stage == "val") and stage != "test":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(
                learned_presentation, inputs["X_holdout"], inputs["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_MAE,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }
