import argparse
import os
from datetime import datetime
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.TimesNet import TimesNet
from models.CECFmodel import CECFmodel
from models.ImputeFormer import ImputeFormerModel
from models.brits import BRITS
from models.saits import SAITS
from models.transformer import TransformerEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import (
    Controller,
    setup_logger,
    save_model,
    load_model,
    check_saving_dir_for_model,
    masked_mae_cal,
    masked_rmse_cal,
    masked_mre_cal,
    str2bool, masked_mse_cal,
)
from utils.unified_dataloader import UnifiedDataLoader

try:
    import nni
except ImportError:
    pass


def set_seed(seed):
    """
    Define as sementes para todas as bibliotecas relevantes para garantir a reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # para multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Define a dictionary for models for easy calling by name
MODEL_DICT = {
    # Self-Attention (SA) based models
    "Transformer": TransformerEncoder,
    "SAITS": SAITS,
    "CECFmodel": CECFmodel,
    "TimesNet": TimesNet,
    "ImputeFormer": ImputeFormerModel,
    # RNN based models
    "BRITS": BRITS,
}
# Define a dictionary for optimizers
OPTIMIZER = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024, help='Random seed')
parser.add_argument("--train_only", action="store_true",
                    help="If set, only run training and skip the automatic test after training. Default: False")

# Basic settings
parser.add_argument("--result_saving_base_dir", type=str, default="results", help="Base directory for saving results")
parser.add_argument("--model_name", type=str, help="Model name, e.g., SAITS, BRITS")
parser.add_argument("--device", type=str, default="cuda", help="Device, 'cuda' or 'cpu'")
parser.add_argument("--test_mode", dest="test_mode", action="store_true", help="Test mode to evaluate a saved model")
parser.add_argument("--param_searching_mode", dest="param_searching_mode", action="store_true",
                    help="Parameter searching mode, using NNI for hyperparameter tuning")

# Dataset related
parser.add_argument("--dataset_path", type=str, help="Path to the dataset file")
parser.add_argument("--seq_len", type=int, default=192, help="Time series sequence length")
parser.add_argument("--feature_num", type=int, help="Number of features")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")

# Training strategy
parser.add_argument("--epochs", type=int, default=70, help="Total number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--optimizer_type", type=str, default="adam", help="Optimizer type, e.g., 'adam', 'adamw'")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay in the optimizer, e.g., 1e-5")
parser.add_argument("--max_norm", type=float, default=0, help="Max norm for gradient clipping, 0 to disable")
parser.add_argument("--early_stop_patience", type=int, default=30, help="Patience for early stopping")
parser.add_argument("--eval_every_n_steps", type=int, default=7, help="Evaluate on the validation set every N steps")
parser.add_argument("--MIT", type=str2bool, default=True, help="Whether to use Masked Imputation Training")
parser.add_argument("--ORT", type=str2bool, default=True, help="Whether to use Observation Reconstruction Training")
parser.add_argument("--use_scheduler", type=str2bool, default=True,
                    help="Whether to enable learning rate scheduler (ReduceLROnPlateau). Default: False")
parser.add_argument("--disable_early_stop_before_n_epochs", type=int, default=30,
                    help="If greater than 0, early stopping will not be triggered until the training reaches the "
                         "specified epoch; 0 means this strategy is not enabled.")

# Loss weights
parser.add_argument("--imputation_loss_weight", type=float, default=1, help="Weight for imputation loss")
parser.add_argument("--reconstruction_loss_weight", type=float, default=1, help="Weight for reconstruction loss")
parser.add_argument("--consistency_loss_weight", type=float, default=1,
                    help="Weight for consistency loss (for BRITS only)")
parser.add_argument("--enhanced_loss_weight", type=float, default=0, help="Weight for enhanced path")
parser.add_argument("--artificial_loss_weight", type=float, default=1, help="Weight for artificial path")
# Model saving and loading
parser.add_argument("--model_saving_strategy", type=str, default="best",
                    help="Model saving strategy: 'best', 'all', 'none'")
parser.add_argument("--model_path", type=str, help="Path to a saved model (for test mode)")
parser.add_argument("--save_imputations", type=str2bool, default=True, help="Whether to save imputation results")
parser.add_argument("--result_saving_path", type=str, help="Path to save test results and imputations")

# RNN model specific parameters
parser.add_argument("--rnn_hidden_size", type=int, help="Hidden size of RNN")

# SA (Self-Attention) model specific parameters
parser.add_argument("--mask_emb_dim", type=int, default=16, help="Mask embedding dimension")
parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--param_sharing_strategy", type=str, default="inner_group",
                    help="Parameter sharing strategy: 'inner_group' or 'between_group'")
parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
parser.add_argument("--d_inner", type=int, default=128, help="Inner feed-forward network dimension")
parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
parser.add_argument("--d_k", type=int, default=64, help="Dimension of Key")
parser.add_argument("--d_v", type=int, default=64, help="Dimension of Value")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--diagonal_attention_mask", type=str2bool, default=False,
                    help="Whether to use a diagonal attention mask")
parser.add_argument("--input_with_mask", type=str2bool, default=True, help="Whether to concatenate input with mask")
parser.add_argument("--low_pass_n", type=int, default=40, help="k of low pass filter")

# SAITS specific parameters
parser.add_argument("--n_groups", type=int, default=2, help="Number of groups")
parser.add_argument("--n_group_inner_layers", type=int, default=1, help="Number of inner layers per group")
parser.add_argument("--begin_order", type=int, default=0)
parser.add_argument('--e_layers', type=int, default=1, help='Number of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='Number of decoder layers')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='Channel independence: 0 for dependent, 1 for independent')

# Inputeformer specific parameters
parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--num_nodes', type=int, default=25)
parser.add_argument('--output_dim', type=int, default=1)
parser.add_argument('--input_embedding_dim', type=int, default=24)
parser.add_argument('--feed_forward_dim', type=int, default=256)
parser.add_argument('--learnable_embedding_dim', type=int, default=80)
parser.add_argument('--num_temporal_heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dim-proj', type=int, default=10)

# TimesNet specific parameters
parser.add_argument('--top_k', type=int, default=3, help="Top k")
parser.add_argument('--d_ffn', type=int, default=32, help="Dimension of FFN")
parser.add_argument('--n_kernels', type=int, default=6, help="Number of kernels")
parser.add_argument('--apply_nonstationary_norm', type=str2bool, default=False)



def summary_write_into_tb(summary_writer, info_dict, step, stage):
    """
    Write metrics from the training/validation process into TensorBoard log files for visualization.

    Parameters:
    - summary_writer: TensorBoard's SummaryWriter instance for logging.
    - info_dict: A dictionary containing various losses and evaluation metrics, such as total_loss, imputation_MAE.
    - step: The current step count (can be iteration or epoch).
    - stage: The current phase, such as "train", "val", "test", to differentiate metrics.
    """
    summary_writer.add_scalar(f"total_loss/{stage}", info_dict["total_loss"], step)
    summary_writer.add_scalar(f"imputation_loss/{stage}", info_dict["imputation_loss"], step)
    summary_writer.add_scalar(f"imputation_MAE/{stage}", info_dict["imputation_MAE"], step)
    summary_writer.add_scalar(f"reconstruction_loss/{stage}", info_dict["reconstruction_loss"], step)


def validate(model, val_iter, summary_writer, training_controller, logger):
    """
    Perform the model validation process, evaluate metrics on the validation set, and write to TensorBoard.

    Parameters:
    - model: The PyTorch model to be evaluated.
    - val_iter: DataLoader iterator for the validation set.
    - summary_writer: TensorBoard SummaryWriter instance.
    - training_controller: An object that controls the training logic (e.g., early stopping, model saving).
    - logger: Logger for recording information.

    Returns:
    - state_dict: A dictionary containing the current validation status, such as step count, whether to save the model, best metrics, etc.
    """
    model.eval()  # Set the model to evaluation mode
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    total_loss_collector, imputation_loss_collector, reconstruction_loss_collector, reconstruction_MAE_collector = [], [], [], []

    with torch.no_grad():  # Disable gradient calculation
        for idx, data in enumerate(val_iter):
            inputs, results = model_processing(data, model, "val")

            # Collect original data, masks, and model imputation results for evaluation
            evalX_collector.append(inputs["X_holdout"])
            evalMask_collector.append(inputs["indicating_mask"])
            imputations_collector.append(results["imputed_data"])

            # Collect various loss and MAE metrics
            total_loss_collector.append(results["total_loss"].data.cpu().numpy())
            reconstruction_MAE_collector.append(results["reconstruction_MAE"].data.cpu().numpy())
            reconstruction_loss_collector.append(results["reconstruction_loss"].data.cpu().numpy())
            imputation_loss_collector.append(results["imputation_loss"].data.cpu().numpy())

        # Concatenate results from all batches to compute overall validation metrics
        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(imputations_collector, evalX_collector, evalMask_collector)

    # Calculate the average of all metrics
    info_dict = {
        "total_loss": np.asarray(total_loss_collector).mean(),
        "reconstruction_loss": np.asarray(reconstruction_loss_collector).mean(),
        "imputation_loss": np.asarray(imputation_loss_collector).mean(),
        "reconstruction_MAE": np.asarray(reconstruction_MAE_collector).mean(),
        "imputation_MAE": imputation_MAE.cpu().numpy().mean(),
    }

    # Call the training controller to check for early stopping/saving the model
    state_dict = training_controller("val", info_dict, logger)
    summary_write_into_tb(summary_writer, info_dict, state_dict["val_step"], "val")

    if args.param_searching_mode:
        nni.report_intermediate_result(info_dict["imputation_MAE"])
        if args.final_epoch or state_dict["should_stop"]:
            nni.report_final_result(state_dict["best_imputation_MAE"])

    # --- CODE MODIFICATION: Model Saving Logic ---
    # Check if the model should be saved based on the strategy
    if state_dict["save_model"] and args.model_saving_strategy == "best":
        # If a best model already exists, remove the old one first
        if hasattr(training_controller, 'best_model_path') and \
                training_controller.best_model_path and \
                os.path.exists(training_controller.best_model_path):
            os.remove(training_controller.best_model_path)

        # Define a new, fixed saving path
        saving_path = os.path.join(args.model_saving, "best_model.pth")

        # Record the new best model path in the controller
        training_controller.best_model_path = saving_path

        save_model(model, optimizer, state_dict, args, saving_path)


    elif state_dict["save_model"] and args.model_saving_strategy == "all":
        # Keep the original logic for the 'all' strategy, saving every improved model
        saving_path = os.path.join(
            args.model_saving,
            "model_trainStep_{}_valStep_{}_imputationMAE_{:.4f}".format(
                state_dict["train_step"],
                state_dict["val_step"],
                info_dict["imputation_MAE"],
            ),
        )
        save_model(model, optimizer, state_dict, args, saving_path)
        logger.info(f"Saved model -> {saving_path}")

    return state_dict, info_dict


def model_processing(data, model, stage, optimizer=None, val_dataloader=None, summary_writer=None,
                     training_controller=None, logger=None, scheduler=None):
    """
    Process data and perform model forward pass (and backward pass if training) based on the stage.

    Parameters:
    - data: A batch of data from the DataLoader.
    - model: The model currently in use.
    - stage: The current stage: "train", "val", or "test".
    - optimizer, val_dataloader, etc.: Auxiliary components required during the training phase.

    Returns:
    - If in training stage, returns a boolean indicating if early stopping occurred.
    - If in validation/test stage, returns the processed inputs and model results.
    """

    if stage == "train":
        optimizer.zero_grad()
        if not args.MIT:  # Non-Masked Imputation Training mode
            if args.model_name in ["BRITS"]:
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = map(
                    lambda x: x.to(args.device), data)
                inputs = {
                    "indices": indices,
                    "forward": {"X": X, "missing_mask": missing_mask, "deltas": deltas},
                    "backward": {"X": back_X, "missing_mask": back_missing_mask, "deltas": back_deltas},
                }
            else:
                indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                inputs = {"indices": indices, "X": X, "missing_mask": missing_mask}

            results = result_processing(model(inputs, stage))
            early_stopping = process_each_training_step(results, optimizer, val_dataloader, training_controller,
                                                        summary_writer, logger, scheduler, model=model)

        else:  # Masked Imputation Training mode
            if args.model_name in ["BRITS"]:
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, indicating_mask = map(
                    lambda x: x.to(args.device), data)
                inputs = {
                    "indices": indices, "X_holdout": X_holdout, "indicating_mask": indicating_mask,
                    "forward": {"X": X, "missing_mask": missing_mask, "deltas": deltas},
                    "backward": {"X": back_X, "missing_mask": back_missing_mask, "deltas": back_deltas},
                }
            else:
                indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
                inputs = {"indices": indices, "X": X, "missing_mask": missing_mask, "X_holdout": X_holdout,
                          "indicating_mask": indicating_mask}

            results = result_processing(model(inputs, stage))
            early_stopping = process_each_training_step(results, optimizer, val_dataloader, training_controller,
                                                        summary_writer, logger, scheduler, model=model)

        return early_stopping, results

    else:  # Validation/Test stage
        if args.model_name in ["BRITS"]:
            indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, indicating_mask = map(
                lambda x: x.to(args.device), data)
            inputs = {
                "indices": indices, "X_holdout": X_holdout, "indicating_mask": indicating_mask,
                "forward": {"X": X, "missing_mask": missing_mask, "deltas": deltas},
                "backward": {"X": back_X, "missing_mask": back_missing_mask, "deltas": back_deltas},
            }
            inputs["missing_mask"] = inputs["forward"]["missing_mask"]
        else:
            indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
            inputs = {"indices": indices, "X": X, "missing_mask": missing_mask, "X_holdout": X_holdout,
                      "indicating_mask": indicating_mask}

        results = result_processing(model(inputs, stage))
        return inputs, results


def train(model, optimizer, train_dataloader, test_dataloader, summary_writer, training_controller, logger, scheduler):
    """
    Execute the complete model training pipeline, including multiple epochs, early stopping, and TensorBoard logging.
    """
    for epoch in range(args.epochs):
        early_stopping = False
        args.final_epoch = (epoch == args.epochs - 1)

        # 打印当前 epoch 开始 ---
        logger.info(f"Epoch [{epoch + 1}/{args.epochs}] started...")

        for idx, data in enumerate(train_dataloader):
            model.train()  # Set the model to training mode
            early_stopping, results = model_processing(
                data, model, "train", optimizer, test_dataloader,
                summary_writer, training_controller, logger, scheduler
            )
            # imputed_data、reconstruction_loss、imputation_loss、reconstruction_MAE、imputation_MAE、total_loss
            training_controller("train", results, logger)

            if early_stopping:
                logger.info(f"Early stopping triggered during epoch {epoch + 1}.")
                break

        if early_stopping:
            break

        training_controller.epoch_num_plus_1()

    logger.info("Finished all epochs.")


def process_each_training_step(results, optimizer, val_dataloader, training_controller, summary_writer, logger,
                               scheduler=None, model=None):
    # 执行一次 controller 增加 step 的时机改为在这里，并把 results 一并传入
    state_dict = training_controller(stage="train", info=results, logger=logger)

    # Backpropagation and optimization
    optimizer.zero_grad()
    results["total_loss"].backward()
    if args.max_norm != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
    optimizer.step()

    # Write to TensorBoard (转换成 python scalars)
    info_for_tb = {
        "total_loss": float(results["total_loss"].item()) if hasattr(results["total_loss"], "item") else float(
            results["total_loss"]),
        "imputation_loss": float(results["imputation_loss"].item()) if hasattr(results["imputation_loss"],
                                                                               "item") else float(
            results["imputation_loss"]),
        "imputation_MAE": float(results["imputation_MAE"].item()) if hasattr(results["imputation_MAE"],
                                                                             "item") else float(
            results.get("imputation_MAE", 0.0)),
        "reconstruction_loss": float(results["reconstruction_loss"].item()) if hasattr(results["reconstruction_loss"],
                                                                                       "item") else float(
            results["reconstruction_loss"]),
    }
    summary_write_into_tb(summary_writer, info_for_tb, state_dict["train_step"], "train")

    # Perform validation at the specified frequency
    if state_dict["train_step"] % args.eval_every_n_steps == 0:
        state_dict_from_val, val_info = validate(model, val_dataloader, summary_writer, training_controller, logger)
        val_metric = val_info["imputation_MAE"]

        prev_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_metric)
            new_lr = optimizer.param_groups[0]['lr']
            summary_writer.add_scalar("learning_rate", new_lr, state_dict["train_step"])
            if abs(new_lr - prev_lr) > 1e-12:
                logger.info(f"[Scheduler] LR changed: {prev_lr:.6e} → {new_lr:.6e} (val_metric={val_metric:.6f})")
        else:
            summary_writer.add_scalar("learning_rate", prev_lr, state_dict["train_step"])

        if state_dict_from_val["should_stop"]:
            logger.info("Early stopping triggered, stopping training...")
            return True

    return False


def result_processing(results):
    # 假设 results 中已有 "imputation_loss" 和 "reconstruction_loss"（tensor）
    # 首先乘上权重（不会改动原始 reference）
    imp_loss = results.get("imputation_loss", 0.0)
    rec_loss = results.get("reconstruction_loss", 0.0)

    if isinstance(imp_loss, torch.Tensor):
        imp_loss = imp_loss * args.imputation_loss_weight
    else:
        imp_loss = torch.tensor(float(imp_loss), device=args.device) * args.imputation_loss_weight

    if isinstance(rec_loss, torch.Tensor):
        rec_loss = rec_loss * args.reconstruction_loss_weight
    else:
        rec_loss = torch.tensor(float(rec_loss), device=args.device) * args.reconstruction_loss_weight

    results["reconstruction_loss"] = rec_loss
    results["imputation_loss"] = imp_loss

    total = torch.tensor(0.0, device=args.device)
    # 用加和（非 in-place）来产生 total
    total_terms = []
    if args.MIT:
        total_terms.append(imp_loss)
    if args.ORT:
        total_terms.append(rec_loss)

    if total_terms:
        # 把 list 的 tensor 相加（第一个 tensor 开始）
        total = total_terms[0]
        for t in total_terms[1:]:
            total = total + t

    results["total_loss"] = total
    return results


def test_trained_model(model, test_dataloader):
    """
    Evaluate the trained model on the complete test set and compute imputation-related metrics.

    Parameters:
    - model: The trained PyTorch model.
    - test_dataloader: DataLoader for the test set.
    """
    logger.info("Starting evaluation on the whole test set...")
    model.eval()

    # 初始化收集器
    evalX_collector, evalMask_collector, imputations_collector, results_collector, idx_collector = [], [], [], [], []

    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            # 调用 model_processing 获取 inputs 和 results
            inputs, results = model_processing(data, model, "test")

            # 收集数据
            evalX_collector.append(inputs["X_holdout"])
            evalMask_collector.append(inputs["indicating_mask"])
            imputations_collector.append(results["imputed_data"])
            results_collector.append(results["imputed_data"])
            idx_collector.append(inputs["indices"])  # 收集 idx，用于排序

        # 合并所有批次的数据
        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        idx_collector = torch.cat(idx_collector)  # 合并 idx 数据

        # 计算各种评估指标
        imputation_MAE = masked_mae_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_RMSE = masked_rmse_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_MRE = masked_mre_cal(imputations_collector, evalX_collector, evalMask_collector)

    # 合并结果并根据 idx 排序
    imputed_test_data = torch.cat(results_collector).cpu().numpy()
    idx_collector = idx_collector.cpu().numpy().reshape(-1)
    ordered_imputed_test_data = imputed_test_data[np.argsort(idx_collector)]  # 按 idx 排序

    # 保存排序后的插补数据
    test_imputations_path = os.path.join(args.result_saving_path, "test_imputations.h5")
    with h5py.File(test_imputations_path, "w") as hf:
        hf.create_dataset("imputed_test_set", data=ordered_imputed_test_data)
    logger.info(f"Done saving test imputed data into {test_imputations_path}.")

    # 记录评估结果
    assessment_metrics = {
        "Imputation MAE on the test set": imputation_MAE,
        "Imputation RMSE on the test set": imputation_RMSE,
        "Imputation MRE on the test set": imputation_MRE,
        "Total trainable parameters": args.total_params,
    }

    # 写入评估结果文件并打印到日志
    with open(os.path.join(args.result_saving_path, "overall_performance_metrics.out"), "w") as f:
        logger.info("Overall performance metrics are as follows:")
        for k, v in assessment_metrics.items():
            logger.info(f"{k}: {v}")
            f.write(k + ":" + str(v))
            f.write("\n")


def impute_all_missing_data(model, train_data, val_data, test_data):
    """
    Use the trained model to impute missing values for all datasets (train/val/test) and save the results.

    Parameters:
    - model: The trained model.
    - train_data, val_data, test_data: The corresponding DataLoaders.
    """
    logger.info("Starting to impute missing data in all datasets...")
    model.eval()
    imputed_data_dict = {}

    with torch.no_grad():
        for dataloader, set_name in zip([train_data, val_data, test_data], ["train", "val", "test"]):
            indices_collector, imputations_collector = [], []
            for idx, data in enumerate(dataloader):
                if args.model_name in ["BRITS"]:
                    indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = map(
                        lambda x: x.to(args.device), data)
                    inputs = {
                        "indices": indices,
                        "forward": {"X": X, "missing_mask": missing_mask, "deltas": deltas},
                        "backward": {"X": back_X, "missing_mask": back_missing_mask, "deltas": back_deltas},
                    }
                else:
                    indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                    inputs = {"indices": indices, "X": X, "missing_mask": missing_mask}

                imputed_data, _ = model.impute(inputs)
                indices_collector.append(indices)
                imputations_collector.append(imputed_data)

            indices = torch.cat(indices_collector).cpu().numpy().reshape(-1)
            imputations = torch.cat(imputations_collector).data.cpu().numpy()
            ordered = imputations[np.argsort(indices)]
            imputed_data_dict[set_name] = ordered

    imputation_saving_path = os.path.join(args.result_saving_path, "imputations.h5")
    with h5py.File(imputation_saving_path, "w") as hf:
        hf.create_dataset("imputed_train_set", data=imputed_data_dict["train"])
        hf.create_dataset("imputed_val_set", data=imputed_data_dict["val"])
        hf.create_dataset("imputed_test_set", data=imputed_data_dict["test"])
    logger.info(f"All imputed data have been saved to {imputation_saving_path}.")


if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed)
    # Process arguments based on model type and running mode (e.g., NNI parameter search)
    if args.model_name in ["Transformer", "SAITS", "CECFmodel", "ImputeFormer", "TimesNet"]:
        assert args.param_sharing_strategy in ["inner_group", "between_group"], \
            "Parameter sharing strategy must be 'inner_group' or 'between_group'"
        dict_args = vars(args)
        if args.param_searching_mode:
            tuner_params = nni.get_next_parameter()
            dict_args.update(tuner_params)
            experiment_id = nni.get_experiment_id()
            trial_id = nni.get_trial_id()
            args.model_name = f"{args.model_name}/{experiment_id}/{trial_id}"
            dict_args["d_k"] = dict_args["d_model"] // dict_args["n_head"]
    elif args.model_name in ["BRITS"]:
        dict_args = vars(args)
        if args.param_searching_mode:
            tuner_params = nni.get_next_parameter()
            dict_args.update(tuner_params)
            experiment_id = nni.get_experiment_id()
            trial_id = nni.get_trial_id()
            args.model_name = f"{args.model_name}/{experiment_id}/{trial_id}"
    else:
        raise ValueError(f"Given model_name {args.model_name} is not in the supported list {MODEL_DICT.keys()}")

    # Parameter validation
    assert args.model_saving_strategy.lower() in ["all", "best",
                                                  "none"], "Model saving strategy must be 'all', 'best', or 'none'"
    if args.model_saving_strategy.lower() == "none":
        args.model_saving_strategy = False
    assert args.optimizer_type in OPTIMIZER.keys(), f"Optimizer type must be in {OPTIMIZER.keys()}"
    assert args.device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"

    # Set up logging and model saving directories
    time_now = datetime.now().__format__("%Y-%m-%d_T%H%M%S")
    args.model_saving, args.log_saving = check_saving_dir_for_model(args, time_now)
    args.result_saving_path = args.model_saving
    os.makedirs(args.result_saving_path, exist_ok=True)
    logger = setup_logger(args.log_saving + "_" + time_now, "w")
    logger.info(f"Args: {args}")
    logger.info(f"Model name: {args.model_name}")

    # Initialize data loader, model, and optimizer
    unified_dataloader = UnifiedDataLoader(args.dataset_path, args.seq_len, args.feature_num, args.model_name,
                                           args.batch_size, args.num_workers, args.MIT)
    model = MODEL_DICT[args.model_name](args)
    args.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of trainable parameters: {args.total_params}")

    if "cuda" in args.device and torch.cuda.is_available():
        model = model.to(args.device)

    # ------------------------------
    # 控制流：test_mode / param_searching_mode / train (默认) -> test (除非 train_only)
    # ------------------------------
    if args.test_mode:
        # Test-only 模式（优先）
        if not getattr(args, "result_saving_path", None):
            if args.model_path and os.path.exists(args.model_path):
                args.result_saving_path = os.path.dirname(args.model_path)
            else:
                args.result_saving_path = "./results"
            os.makedirs(args.result_saving_path, exist_ok=True)
            logger.info(f"result_saving_path set to: {args.result_saving_path}")
        logger.info("Entering testing mode (test_mode=True)...")
        os.makedirs(args.result_saving_path, exist_ok=True)
        model = load_model(model, args.model_path, logger)
        test_dataloader = unified_dataloader.get_test_dataloader()
        test_trained_model(model, test_dataloader)
        if args.save_imputations:
            train_data, val_data, test_data = unified_dataloader.prepare_all_data_for_imputation()
            impute_all_missing_data(model, train_data, val_data, test_data)

    elif args.param_searching_mode:
        # 超参数搜索流程（保持原有逻辑）
        logger.info("Entering parameter searching mode...")
        # 这里假设你在 param_searching_mode 下会走训练/验证的流程，按你原有代码组织
        optimizer = OPTIMIZER[args.optimizer_type](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=5e-5)
        train_dataloader, val_dataloader = unified_dataloader.get_train_val_dataloader()
        training_controller = Controller(args.early_stop_patience, args.disable_early_stop_before_n_epochs)
        training_controller.best_model_path = None
        tb_summary_writer = SummaryWriter(os.path.join(args.log_saving, "tensorboard_" + time_now))
        train(model, optimizer, train_dataloader, val_dataloader, tb_summary_writer, training_controller, logger,
              scheduler)
        # param_searching_mode 下是否自动测试按你的需求决定；这里保留不自动测试的默认行为。

    else:
        # 默认行为：先训练，训练结束后自动进入测试（除非 --train_only）
        logger.info("Entering training mode (default behavior)...")
        optimizer = OPTIMIZER[args.optimizer_type](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
        )

        train_dataloader, val_dataloader = unified_dataloader.get_train_val_dataloader()
        training_controller = Controller(args.early_stop_patience, args.disable_early_stop_before_n_epochs)
        training_controller.best_model_path = None

        tb_summary_writer = SummaryWriter(os.path.join(args.log_saving, "tensorboard_" + time_now))
        train(model, optimizer, train_dataloader, val_dataloader, tb_summary_writer, training_controller, logger,
              scheduler)

        # 如果用户指定只训练（--train_only），则训练完直接退出
        if args.train_only:
            logger.info("Training finished and --train_only set -> exiting without running test.")
        else:
            # 否则训练后接着跑测试：优先载入训练中保存的 best model（如果存在），否则用当前模型参数
            logger.info("Training finished -> entering test mode automatically.")
            test_dataloader = unified_dataloader.get_test_dataloader()

            # 如果训练过程保存了 best_model_path，就优先载入（通常在 validate 中保存）
            best_model_path = getattr(training_controller, "best_model_path", None)
            if best_model_path and os.path.exists(best_model_path):
                logger.info(f"Loading best model from training: {best_model_path}")
                model = load_model(model, best_model_path, logger)
            else:
                logger.info("No best_model_path found after training, using current model weights for testing.")

            test_trained_model(model, test_dataloader)
            if args.save_imputations:
                train_data, val_data, test_data = unified_dataloader.prepare_all_data_for_imputation()
                impute_all_missing_data(model, train_data, val_data, test_data)

    logger.info("All tasks done.")
