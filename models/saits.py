from layers.atten import *
from utils.utils import masked_mae_cal


class SAITS(nn.Module):
    def __init__(
        self,
        configs,
    ):
        super().__init__()
        self.n_groups = configs.n_groups
        self.n_group_inner_layers = configs.n_group_inner_layers
        self.input_with_mask = configs.input_with_mask
        self.feature_num = configs.feature_num
        self.actual_feature_num = self.feature_num * 2 if self.input_with_mask else self.feature_num
        self.param_sharing_strategy = configs.param_sharing_strategy
        self.MIT = configs.MIT
        self.device = configs.device
        self.diagonal_attention_mask = configs.diagonal_attention_mask
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_inner = configs.d_inner
        self.n_head = configs.n_head
        self.d_k = configs.d_k
        self.d_v = configs.d_v
        self.dropout = configs.dropout

        if self.param_sharing_strategy == "between_group":
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
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
                        self.dropout,
                        0,
                    )
                    for _ in range(self.n_group_inner_layers)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList([
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
                        self.dropout,
                        0,
                    )
                    for _ in range(self.n_group_inner_layers)
            ])
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList([
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
                        self.dropout,
                        0,
                    )
                    for _ in range(self.n_groups)
                ])
            self.layer_stack_for_second_block = nn.ModuleList([
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
                        self.dropout,
                        0,
                    )
                    for _ in range(self.n_groups)
            ])

        self.dropout = nn.Dropout(p=self.dropout)
        self.position_enc = PositionalEncoding(self.d_model, n_position=self.seq_len)
        # for the 1st block
        self.embedding_1 = nn.Linear(self.actual_feature_num, self.d_model)
        self.reduce_dim_z = nn.Linear(self.d_model, self.feature_num)
        # for the 2nd block
        self.embedding_2 = nn.Linear(self.actual_feature_num, self.d_model)
        self.reduce_dim_beta = nn.Linear(self.d_model, self.feature_num)
        self.reduce_dim_gamma = nn.Linear(self.feature_num, self.feature_num)
        # for the 3rd block
        self.weight_combine = nn.Linear(self.feature_num + self.seq_len, self.feature_num)

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]
        # the first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(self.position_enc(input_X_for_first))
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # the second DMSA block
        input_X_for_second = (
            torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        )
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # the attention-weighted combination block
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # replace non-missing part with original data
        X_c = masks * X + (1 - masks) * X_tilde_3

        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, stage):
        X, masks = inputs["X"], inputs["missing_mask"]
        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)

        reconstruction_loss += masked_mae_cal(X_tilde_1, X, masks)
        reconstruction_loss += masked_mae_cal(X_tilde_2, X, masks)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X, masks)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        if (self.MIT or stage == "val") and stage != "test":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(
                X_tilde_3, inputs["X_holdout"], inputs["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_loss,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": final_reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }
