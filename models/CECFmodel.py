from layers.ResampleBlock import MultiScaleStochasticResampler, MultiScaleResampleFusion
from layers.MissingEnhencedBlock import MissingInfoEnhenced, LinearTrunkGeneration
from layers.Normalization import nonstationary_norm, nonstationary_denorm
from layers.atten import *
from utils.utils import masked_mae_cal


class CECFmodel(nn.Module):
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
        self.fuse_linear = nn.Linear(self.d_model, self.d_model)
        self.fuse_linear_z = nn.Linear(self.feature_num * 3, self.d_model)
        self.read_out = nn.Linear(self.d_model, self.feature_num)
        self.apply_nonstationary_norm = configs.apply_nonstationary_norm
        self.position_enc = PositionalEncoding(self.d_model, n_position=self.seq_len)
        self.missing_enhencd = MissingInfoEnhenced(self.feature_num, num_freq=5, max_time=self.seq_len)
        self.bayes_resampler = MultiScaleStochasticResampler(
            n_scales=3,
            d_model=self.feature_num,
            min_scale=2,
            max_scale=8,
            mode='linear'
        )
        self.resample_fusion = MultiScaleResampleFusion(d_model=self.d_model,feature_num=self.feature_num,
                                                        n_scales=3)
        self.linear_trunk = nn.Linear(self.feature_num, self.d_model)
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
        self.enhanced_loss_weight = configs.enhanced_loss_weight
        self.artificial_loss_weight = configs.artificial_loss_weight

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            X, means, stdev = nonstationary_norm(X, masks)

        trunk, lov, nov = LinearTrunkGeneration(X, masks)
        X_stem = self.missing_enhencd(trunk, lov, nov, masks)

        X_resampled_list = self.bayes_resampler(X_stem)
        X_fused = self.resample_fusion(X_stem, X_resampled_list)
        # X_fused = self.linear_trunk(X_stem)

        enc_output = self.position_enc(X_fused)
        time_points = self.layer_norm(enc_output)

        for encoder_layer in self.layer_stack:
            enc_output, _ = encoder_layer(enc_output, time_points)
        reconstruction = self.read_out(enc_output)

        if self.apply_nonstationary_norm:
            reconstruction = nonstationary_denorm(reconstruction, means, stdev)

        X_c = X * masks + reconstruction * (1 - masks)

        return X_c, [X, reconstruction]

    def forward(self, inputs, stage):
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, [enhenced_data, reconstruction] = self.impute(inputs)

        reconstruction_MAE = masked_mae_cal(reconstruction, X, masks)

        if (self.MIT or stage == "val") and stage != "test":
            imputation_MAE = self.artificial_loss_weight * masked_mae_cal(
                imputed_data, inputs.get("X_holdout"), inputs.get("indicating_mask")
            )
            imputation_MAE += self.enhanced_loss_weight * masked_mae_cal(
                enhenced_data, inputs.get("X_holdout"), inputs.get("indicating_mask")
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
