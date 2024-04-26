import importlib
import importlib.util
from pathlib import Path

import torch
from cprint import *


class ModelBuilder:
    def __init__(self, model_configs: dict, training_config: dict):
        self.training_config = training_config
        model_field = model_configs.get('model_field')
        self.picked_model_config = model_configs.get(model_field)
        model_filepath = self.picked_model_config.get("model_filepath")
        model_class = self.picked_model_config.get("model_class")
        self.model = self.load_model_from_path(filepath=Path(model_filepath),
                                               class_name=model_class)
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

    def load_model_from_path(self, filepath: Path, class_name: str):
        module = self.load_module(filepath)
        model_obj = getattr(module, class_name)
        # only works if the constructor takes a single argument
        try:
            return model_obj(self.picked_model_config)
        except TypeError as e:
            hint = ("The specific model config is passed to the constructor as an argument. The model needs"
                    " to take a single argument in the constructor.")
            raise TypeError(f"{e}\n{hint}")

    @staticmethod
    def load_module(filepath: Path):
        module_name = filepath.name
        spec = importlib.util.spec_from_file_location(module_name, filepath.resolve())
        if spec is not None:
            _spec = spec
            module = importlib.util.module_from_spec(_spec)
            sys.modules[module_name] = module
            assert _spec.loader is not None
            loader = _spec.loader
            loader.exec_module(module)
            return module
        raise ImportError(f"Couldn't find a module under: {filepath}")

    def get_model(self):
        return self.model

    def get_model_config(self):
        return self.picked_model_config
