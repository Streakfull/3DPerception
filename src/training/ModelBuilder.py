import importlib
import importlib.util
from pathlib import Path

import torch
from cprint import *

from src.training.ModuleLoader import load_module_from_path


class ModelBuilder:
    def __init__(self, model_configs: dict, training_config: dict):
        self.training_config = training_config
        model_field = model_configs.get('model_field')
        self.picked_model_config = model_configs.get(model_field)
        model_filepath = self.picked_model_config.get("model_filepath")
        model_class = self.picked_model_config.get("model_class")
        model_type = load_module_from_path(filepath=Path(model_filepath),
                                           class_name=model_class)
        self.model = model_type(self.picked_model_config)
        self.model_to_device()
        self.load_model_ckpt()

    def model_to_device(self):
        if torch.cuda.is_available():
            torch.cuda.mem_get_info()
        device = self.training_config["device"]
        if "cpu" == device:
            cprint.warn('Using CPU')
        else:
            cprint.ok('Using device:', device)

        self.model.to(device)

    def load_model_ckpt(self):
        if self.training_config["load_ckpt"]:
            self.model.load_ckpt(self.training_config['ckpt_path'])

    def get_model(self):
        return self.model

    def get_model_config(self):
        return self.picked_model_config
