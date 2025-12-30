import h5py
import numpy as np
import torch
from utils.utils import masked_mae_cal, masked_mre_cal, masked_rmse_cal

dataset_dir = "generated_datasets/carbon-NamCo_mr025_ft02"
testfile_dir = "results/ImputeFormer/models/2025-12-09_T134111"

dataset_path = dataset_dir + "/datasets.h5"
testfile_path = testfile_dir + "/test_imputations.h5"

# ==== load ====
with h5py.File(testfile_path, "r") as f:
    imputed = f["imputed_test_set"][:]     # (107,192,25)

with h5py.File(dataset_path, "r") as f:
    X_hat  = f["test/X"][:]            # (107,192,25)
    mask   = f["test/indicating_mask"][:]  # (107,192,25)


# === 取最后一维 ===
imputed_last = imputed[..., -1]    # (107,192)
X_hat_last   = X_hat[..., -1]      # (107,192)
mask_last    = mask[..., -1]       # (107,192)


# === reshape 为一维 ===
imputed_flat = imputed_last.reshape(-1)   # (107*192,)
X_hat_flat   = X_hat_last.reshape(-1)
mask_flat    = mask_last.reshape(-1)

# print(mask_flat.sum())
# print(mask_flat.mean())
# print(np.isnan(imputed_flat).sum())
# print(np.isnan(X_hat_flat).sum())

imputed_flat = torch.from_numpy(imputed_flat)
X_hat_flat = torch.from_numpy(np.nan_to_num(X_hat_flat)).float()
mask_flat    = torch.from_numpy(mask_flat)

# print(imputed_flat.shape)
# print(X_hat_flat.shape)
# print(mask_flat.shape)

imputation_MAE = masked_mae_cal(imputed_flat, X_hat_flat, mask_flat)
imputation_RMSE = masked_rmse_cal(imputed_flat, X_hat_flat, mask_flat)
imputation_MRE = masked_mre_cal(imputed_flat, X_hat_flat, mask_flat)

print(imputation_MAE, imputation_RMSE, imputation_MRE)

