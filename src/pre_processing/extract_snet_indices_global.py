from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_normal
import yaml
import torch
from typing import Type
import numpy as np
from cprint import *
from einops import rearrange

from src.datasets.base_dataset import BaseDataSet
from src.training.DataLoaderHandler import DataLoaderHandler
from src.training.ModelBuilder import ModelBuilder


class ExtractSnetIndices:

    def __init__(self, dataset_type: Type[BaseDataSet], configs_path="src/configs/global_configs.yaml", options: dict = {}):
        self.dataset_type = dataset_type
        self.codebook_output_path = "src/weights/codebook.pth"
        self.options = options
        self.tqdm = tqdm_notebook if self.options.get(
            "tdm_notebook", False) else tqdm_normal
        with open(configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
        self.training_config = self.global_configs["training"]
        self.data_loader_handler = DataLoaderHandler(global_configs=self.global_configs,
                                                     batch_size=self.training_config['batch_size'],
                                                     test_size=self.training_config['test_size'],
                                                     num_workers=self.training_config['num_workers'])
        self.train_dataloader, self.validation_dataloader = self.data_loader_handler.get_dataloaders()
        assert self.global_configs["model"]["model_field"] == "pvqvae" or self.global_configs[
            "model"]["model_field"] == "globalPVQVAE", "pvqvae or globalpvqvae model required to extract codes"
        assert self.global_configs["training"]["load_ckpt"], "Load checkpoint is not set to true"
        self.model = ModelBuilder(
            self.global_configs["model"], self.training_config).get_model()
        self._save_codebook_weights()
        self._set_device()

    def _save_codebook_weights(self):
        codebook_weights = {
            'codebook': self.model.get_codebook_weight()
        }

        torch.save(codebook_weights, self.codebook_output_path)
        cprint.ok(f"{self.codebook_output_path} saved")

    def extract(self):
        self._extract_dataloader(self.train_dataloader)
        self._extract_dataloader(self.validation_dataloader)

    def extract_train(self):
        self._extract_dataloader(self.train_dataloader)

    def extract_validation(self):
        self._extract_dataloader(self.validation_dataloader)

    def _extract_dataloader(self, dl):
        for index, batch in self.tqdm(enumerate(dl), total=len(dl)):
            with torch.no_grad():
                self.dataset_type.move_batch_to_device(
                    batch, self.device)
                sdf = self.model.get_batch_input(batch)
                self.model.inference(sdf)
                _, z_q, info = self.model.x_recon, self.model.zq_cubes, self.model.info
                z_q_indices = info[-1]
                batch_size = z_q.shape[0]
                d, h, w = z_q.shape[-3:]
                # z_q_indices = rearrange(
                #     z_q_indices.squeeze(), '(b d h w) -> b d h w', b=batch_size, d=d, h=h, w=w)
                out = self.model.decode_from_quant(z_q_indices)
                z_q = z_q.detach().cpu().numpy()
                z_q_indices = z_q_indices.detach().cpu().numpy()
                for i in range(batch_size):
                    z_q_i = z_q[i]
                    z_q_indices_i = z_q_indices[i]
                    base_path = batch["path"][i]
                    np.save(f"{base_path}/global_code_gan.npy", z_q_i)
                    np.save(f"{base_path}/global_codeix_gan.npy",
                            z_q_indices_i)

    def _set_device(self):
        self.device = torch.device('cpu')
        if torch.cuda.is_available() and self.training_config['device'].startswith('cuda'):
            self.device = torch.device(self.training_config['device'])
            cprint.ok('Using device:', self.training_config['device'])
        else:
            cprint.warn('Using CPU')
