"""
Extended dataset wrappers for BenchPOTS and CSDI PM25 datasets.
These require optional dependencies: pip install pypots benchpots pygrinder tsdb
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')


class BenchPOTSWrapper(Dataset):
    """
    Wrapper for BenchPOTS datasets (PhysioNet2012, PEMS03, etc.).
    Returns (X, X_ori, mask, indicating_mask, x_mark, y_mark) tuples.
    """

    def __init__(self, dataset_name, subset='train', root_path='../dataset/benchpots',
                 mit_rate=0.2, benchpots_missing_rate=0.1, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.root_path = root_path
        self.mit_rate = mit_rate
        self.benchpots_missing_rate = benchpots_missing_rate
        self._load_data()

    def _load_data(self):
        try:
            from benchpots import datasets as bp_datasets
        except ImportError:
            raise ImportError(
                "BenchPOTS is not installed. Install with: pip install benchpots pygrinder tsdb"
            )

        print(f"Loading {self.dataset_name} from BenchPOTS...")

        preprocess_fn = getattr(bp_datasets, f'preprocess_{self.dataset_name}', None)
        if preprocess_fn is None:
            raise ValueError(
                f"Unknown BenchPOTS dataset: {self.dataset_name}. "
                f"Check benchpots.datasets for available preprocessing functions."
            )

        # PhysioNet2012 requires subset arg
        if self.dataset_name == 'physionet2012':
            data = preprocess_fn(subset='set-a', rate=self.benchpots_missing_rate)
        else:
            data = preprocess_fn(rate=self.benchpots_missing_rate)

        key = self.subset
        self.X = torch.FloatTensor(data[f"{key}_X"])

        if f"{key}_X_ori" in data:
            self.X_ori = torch.FloatTensor(data[f"{key}_X_ori"])
        else:
            self.X_ori = self.X.clone()

        self.original_missing_mask = (~torch.isnan(self.X)).float()
        self.X = torch.nan_to_num(self.X, nan=0.0)
        self.X_ori = torch.nan_to_num(self.X_ori, nan=0.0)

        self.n_samples, self.n_steps, self.n_features = self.X.shape
        print(f"  {self.dataset_name} {self.subset}: shape={self.X.shape}, "
              f"missing={1 - self.original_missing_mask.mean().item():.2%}")

    def _generate_mit_mask(self, X, original_mask):
        observed_indices = original_mask.bool()
        n_observed = observed_indices.sum()
        n_to_mask = int(n_observed * self.mit_rate)

        if n_to_mask > 0:
            observed_flat = torch.where(observed_indices.flatten())[0]
            mask_indices = observed_flat[torch.randperm(len(observed_flat))[:n_to_mask]]
            indicating_mask = torch.zeros_like(X).flatten()
            indicating_mask[mask_indices] = 1
            indicating_mask = indicating_mask.reshape(X.shape)
            mit_mask = original_mask - indicating_mask
        else:
            indicating_mask = torch.zeros_like(X)
            mit_mask = original_mask

        return mit_mask, indicating_mask

    def __getitem__(self, index):
        X_sample = self.X[index]
        X_ori_sample = self.X_ori[index]
        original_mask = self.original_missing_mask[index]

        if self.subset == 'train':
            mit_mask, indicating_mask = self._generate_mit_mask(X_ori_sample, original_mask)
            X_mit = X_ori_sample.clone()
            X_mit[indicating_mask.bool()] = 0
        else:
            X_mit = X_sample
            mit_mask = original_mask
            indicating_mask = torch.zeros_like(original_mask)

        seq_x_mark = torch.zeros(self.n_steps, 4)
        seq_y_mark = torch.zeros(self.n_steps, 4)

        return X_mit, X_ori_sample, mit_mask, indicating_mask, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_samples


class PM25Wrapper(Dataset):
    """
    Wrapper for CSDI PM25 dataset (36 monitoring stations).
    Returns (X, X_ori, mask, indicating_mask, x_mark, y_mark) tuples.
    """

    def __init__(self, subset='train', root_path='../dataset/pm25',
                 eval_length=36, target_dim=36, validindex=0, **kwargs):
        super().__init__()
        self.subset = subset
        self.root_path = root_path
        self.eval_length = eval_length
        self.target_dim = target_dim
        self.validindex = validindex
        self._load_data()

    def _load_data(self):
        import pickle
        import pandas as pd

        meanstd_path = os.path.join(self.root_path, "pm25_meanstd.pk")
        with open(meanstd_path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)

        if self.subset == "train":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            month_list.pop(self.validindex)
        elif self.subset == "val":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            month_list = month_list[self.validindex:self.validindex + 1]
        elif self.subset == "test":
            month_list = [3, 6, 9, 12]

        ground_path = os.path.join(self.root_path, "Code/STMVL/SampleData/pm25_ground.txt")
        missing_path = os.path.join(self.root_path, "Code/STMVL/SampleData/pm25_missing.txt")

        df_ground = pd.read_csv(ground_path, index_col="datetime", parse_dates=True)
        df_missing = pd.read_csv(missing_path, index_col="datetime", parse_dates=True)

        all_data, all_data_ori, all_masks, all_indicating_masks = [], [], [], []

        for month in month_list:
            month_ground = df_ground[df_ground.index.month == month]
            month_missing = df_missing[df_missing.index.month == month]

            ground_values = month_ground.values
            missing_values = month_missing.values

            ground_mask = ~np.isnan(ground_values)
            missing_mask = ~np.isnan(missing_values)
            indicating_mask = ground_mask & ~missing_mask

            ground_normalized = (ground_values - self.train_mean) / self.train_std
            missing_normalized = (missing_values - self.train_mean) / self.train_std

            n_windows = len(month_ground) - self.eval_length + 1
            for i in range(max(n_windows, 0)):
                all_data.append(missing_normalized[i:i + self.eval_length])
                all_data_ori.append(ground_normalized[i:i + self.eval_length])
                all_masks.append(missing_mask[i:i + self.eval_length])
                all_indicating_masks.append(indicating_mask[i:i + self.eval_length])

        self.X = torch.FloatTensor(np.array(all_data))
        self.X_ori = torch.FloatTensor(np.array(all_data_ori))
        self.missing_mask = torch.FloatTensor(np.array(all_masks))
        self.indicating_mask = torch.FloatTensor(np.array(all_indicating_masks))

        self.X = torch.nan_to_num(self.X, nan=0.0)
        self.X_ori = torch.nan_to_num(self.X_ori, nan=0.0)

        self.n_samples, self.n_steps, self.n_features = self.X.shape
        print(f"  PM25 {self.subset}: shape={self.X.shape}, "
              f"missing={1 - self.missing_mask.mean().item():.2%}")

    def __getitem__(self, index):
        X_sample = self.X[index]
        X_ori_sample = self.X_ori[index]
        mask_sample = self.missing_mask[index]
        indicating_sample = self.indicating_mask[index]

        seq_x_mark = torch.zeros(self.n_steps, 4)
        seq_y_mark = torch.zeros(self.n_steps, 4)

        return X_sample, X_ori_sample, mask_sample, indicating_sample, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_samples
