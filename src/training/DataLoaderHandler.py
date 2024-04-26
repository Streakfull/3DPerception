from typing import Type

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from src.datasets.base_dataset import BaseDataSet


class DataLoaderHandler:
    def __init__(self, global_configs: DictConfig, dataset_type: Type[BaseDataSet], batch_size: int,
                 num_workers: int = 1, test_size: float = 0.2):
        global_dataset_config = global_configs["dataset"]
        dataset_field = global_dataset_config["dataset_field"]
        local_dataset_config = global_dataset_config[dataset_field]
        dataset = dataset_type(global_dataset_config, local_dataset_config)
        print('length: ', len(dataset))

        train_ds, valid_ds = self.train_val_dataset(dataset, test_size)

        self.train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,  # Data is usually loaded in parallel by num_workers
            pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        )

        self.validation_dataloader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,  # Data is usually loaded in parallel by num_workers
            pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        )

    @staticmethod
    def train_val_dataset(dataset, test_size):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_size)

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        return train_ds, val_ds

    def get_dataloaders(self):
        return self.train_dataloader, self.validation_dataloader
