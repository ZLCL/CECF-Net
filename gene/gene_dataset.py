import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import chardet
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump
from utils.utils import setup_logger
from data_processing_utils import (
    window_truncate,
    random_mask,
    add_artificial_mask,
    saving_into_h5,
)


def read_csv_safely(file_path, index_col=None):
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read(10000))['encoding']
    print(f"Detected encoding: {encoding}")
    return pd.read_csv(file_path, index_col=index_col, encoding=encoding)


def filter_missing_ratio(data, threshold=0.2):
    """
    过滤掉缺失率超过 threshold 的样本
    data: np.ndarray, shape (num_samples, seq_len, num_features)
    """
    if data.size == 0:
        return data
    # 每个样本的缺失率
    missing_ratio = np.isnan(data).sum(axis=(1, 2)) / (data.shape[1] * data.shape[2])
    # 只保留缺失率 <= threshold 的样本
    keep_mask = missing_ratio <= threshold
    return data[keep_mask]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("--file_path", help="path of dataset file", type=str)
    parser.add_argument("--fault_tolerance", type=float, default=0.4, help="fault tolerance")
    parser.add_argument(
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )
    parser.add_argument("--seq_len", help="sequence length", type=int, default=24)
    parser.add_argument("--sliding_len", help="sequence length", type=int, default=12)
    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--saving_path", type=str, help="parent dir of generated dataset", default="."
    )
    parser.add_argument("--index_col", type=str, help="index colum name", default="date")
    args = parser.parse_args()

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    logger = setup_logger(
        os.path.join(dataset_saving_dir + "/dataset_generating.log"),
        "Generate dataset",
        mode="w",
    )
    logger.info(args)

    df = read_csv_safely(args.file_path, args.index_col)
    # df = pd.read_csv(args.file_path, index_col=args.index_col)
    df.index = pd.to_datetime(df.index)
    feature_names = df.columns.tolist()
    feature_num = len(feature_names)
    df["datetime"] = pd.to_datetime(df.index)

    # === 新的时间顺序划分逻辑 ===
    unique_months = sorted(df["datetime"].dt.to_period("M").unique())

    n_months = len(unique_months)
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    # 计算每个区间的切分点（四舍五入以避免整数问题）
    n_train = int(n_months * train_ratio)
    n_val = int(n_months * val_ratio)
    n_test = n_months - n_train - n_val

    selected_as_train = unique_months[:n_train]
    selected_as_val = unique_months[n_train:n_train + n_val]
    selected_as_test = unique_months[n_train + n_val:]

    logger.info(f"Months used for training: {selected_as_train}")
    logger.info(f"Months used for validation: {selected_as_val}")
    logger.info(f"Months used for testing: {selected_as_test}")

    # === 实际切分数据 ===
    train_set = df[df["datetime"].dt.to_period("M").isin(selected_as_train)]
    val_set = df[df["datetime"].dt.to_period("M").isin(selected_as_val)]
    test_set = df[df["datetime"].dt.to_period("M").isin(selected_as_test)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len, args.sliding_len)
    val_set_X = window_truncate(val_set_X, args.seq_len, args.sliding_len)
    test_set_X = window_truncate(test_set_X, args.seq_len, args.sliding_len)

    # 过滤掉缺失率超过 40% 的样本
    train_set_X = filter_missing_ratio(train_set_X, threshold=args.fault_tolerance)
    val_set_X = filter_missing_ratio(val_set_X, threshold=args.fault_tolerance)
    test_set_X = filter_missing_ratio(test_set_X, threshold=args.fault_tolerance)
    logger.info(
        f"After filtering: train={train_set_X.shape[0]}, val={val_set_X.shape[0]}, test={test_set_X.shape[0]}"
    )

    # add missing values in train set manually
    if args.artificial_missing_rate > 0:
        train_set_X_shape = train_set_X.shape
        train_set_X = train_set_X.reshape(-1)
        indices = random_mask(train_set_X, args.artificial_missing_rate)
        train_set_X[indices] = np.nan
        train_set_X = train_set_X.reshape(train_set_X_shape)
        logger.info(
            f"Already masked out {args.artificial_missing_rate * 100}% values in train set"
        )

    train_set_dict = add_artificial_mask(
        train_set_X, args.artificial_missing_rate, "train"
    )
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(
        test_set_X, args.artificial_missing_rate, "test"
    )
    logger.info(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
    }
    train_sample_num = len(train_set_dict["X"])
    val_sample_num = len(val_set_dict["X"])
    test_sample_num = len(test_set_dict["X"])
    total_sample_num = train_sample_num + val_sample_num + test_sample_num
    logger.info(
        f"Feature num: {feature_num},\n"
        f"{train_sample_num} ({(train_sample_num / total_sample_num):.3f}) samples in train set\n"
        f"{val_sample_num} ({(val_sample_num / total_sample_num):.3f}) samples in val set\n"
        f"{test_sample_num} ({(test_sample_num / total_sample_num):.3f}) samples in test set\n"
    )

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    pickle_dump(scaler, os.path.join(dataset_saving_dir, "scaler"))
    logger.info(f"All done. Saved to {dataset_saving_dir}.")
