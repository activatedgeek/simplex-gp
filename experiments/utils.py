from typing import List, Union
from pathlib import Path
from torch.utils.data import Dataset
import torch
from scipy.io import loadmat
import itertools
import os
import random
import numpy as np
import wandb


def set_seeds(seed=None):
  if seed:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize(train_x, train_y, test_x, test_y):
    x_mean = train_x.mean(0, keepdim=True)
    x_std = train_x.std(0, keepdim=True) + 1e-6

    y_mean = train_y.mean(0, keepdim=True)
    y_std = train_y.std(0, keepdim=True) + 1e-6

    train_x = (train_x - x_mean) / x_std
    train_y = (train_y - y_mean) / y_std

    test_x = (test_x - x_mean) / x_std
    test_y = (test_y - y_mean) / y_std    

    return train_x, train_y, test_x, test_y


class UCIDataset(Dataset):
    UCI_PATH = Path(os.path.expanduser("~/datasets/uci/"))

    def __init__(
        self,
        dataset_path: Union[Path, str],
        mode: str = "train",
        dtype=torch.float32,
        device="cpu",
    ):
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_path.stem
        data = loadmat(str(dataset_path))["data"]
        data = torch.as_tensor(data, dtype=dtype, device=device)

        # make train/val/test split
        N = data.size(0)
        n_train_val = int(0.8 * N)
        n_train = int(0.8 * n_train_val)

        train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
        val_x, val_y = data[n_train:n_train_val, :-1], data[n_train:n_train_val, -1]
        test_x, test_y = data[n_train_val:, :-1], data[n_train_val:, -1]

        if mode == "train":
            self.x, self.y = train_x, train_y
        elif mode == "val":
            self.x, self.y, = val_x, val_y
        elif mode == "test":
            self.x, self.y = test_x, test_y
        else:
            raise ValueError("mode must be one of 'train', 'val', or 'test'")

    def __getitem__(self, index):
        return (self.x.__getitem__(index), self.y.__getitem__(index))

    def __len__(self):
        return len(self.y)

    @staticmethod
    def all_dataset_paths(uci_data_dir: Path = UCI_PATH):
        return list(uci_data_dir.glob("*.mat"))

    @staticmethod
    def all_dataset_names(uci_data_dir: Path = UCI_PATH):
        return [p.stem for p in UCIDataset.all_dataset_paths(uci_data_dir)]

    @staticmethod
    def create(*names: Union[str, list], uci_data_dir: Path = None, mode="train", dtype=torch.float32, device="cpu") -> List:
        """Create one or more `UCIDataset`s from their names

        `names` can be the name of a group of datasets as listed below

        Example:
            ```
            UCIDataset.create("challenger")
            UCIDataset.create("challenger", "fertility")
            UCIDataset.create("small")
            ```

        """
        uci_data_dir = Path(uci_data_dir) or UCI_PATH

        def get(dataset_names: List[str]):
            return [(uci_data_dir / d / d).with_suffix(".mat") for d in dataset_names]

        groups = {
            **{
                "all": UCIDataset.all_dataset_paths(uci_data_dir),
                "small": get(
                    [
                        "challenger",
                        "fertility",
                        "concreteslump",
                        "autos",
                        "servo",
                        "breastcancer",
                        "machine",
                        "yacht",
                        "autompg",
                        "housing",
                        "forest",
                        "stock",
                        "pendulum",
                        "energy",
                        "concrete",
                        "solar",
                        "airfoil",
                        "wine",
                        "gas",
                        "skillcraft",
                        "sml",
                        "parkinsons",
                        "pumadyn32nm",
                    ]
                ),
                "medium": get(
                    [
                        "pol",
                        "elevators",
                        "bike",
                        "kin40k",
                        "protein",
                        "keggdirected",
                        "slice",
                        "keggundirected",
                    ]
                ),
                "large": get(["3droad", "song", "buzz"]),
                "huge": get(["houseelectric"]),
            },
            # Allow individual dataset names
            **{p.stem: [p] for p in UCIDataset.all_dataset_paths(uci_data_dir)},
        }
        datasets = itertools.chain.from_iterable([groups[g_or_n] for g_or_n in names])
        datasets = [UCIDataset(dataset_path, mode=mode, device=device, dtype=dtype) for dataset_path in datasets]
        if len(datasets) == 1:
            return datasets[0]
        else:
            return datasets
