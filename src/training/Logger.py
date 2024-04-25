import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from cprint import *
from torch.utils.tensorboard import SummaryWriter

from src.utils.util import mkdir


@dataclass
class TrainingVariables:
    def __init__(self, experiment_dir, train_loss_running=0., best_loss_val=np.inf, start_iteration=0, last_loss=0.):
        self.train_loss_running = train_loss_running
        self.best_loss_val = best_loss_val
        self.start_iteration = start_iteration
        tb_dir = f"{experiment_dir}/tb"
        self.writer = SummaryWriter(log_dir=tb_dir)
        self.model_checkpoint_path = f"{experiment_dir}/checkpoints"
        self.loss_log_name = f"{experiment_dir}/loss_log.txt"
        self.visuals_path = f"{experiment_dir}/visuals"
        self.last_loss = last_loss


class Logger:
    def __init__(self, training_config):
        today = time.strftime("%Y-%m-%d")
        cprint.ok(training_config)
        description = training_config["description"]  # Describe Experiment params here
        logs_dir = training_config["logs_dir"]
        mkdir(logs_dir)
        self.experiment_dir = Path(logs_dir) / training_config['name'] / datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S")
        mkdir(self.experiment_dir)
        loss_log_title = "Loss Log " + today

        with open(f"{self.experiment_dir}/description.txt", "w") as file1:
            file1.write(description)

        with open(f"{self.experiment_dir}/global_configs.json", "w") as file1:
            json_object = json.dumps(training_config, indent=4)
            file1.write(str(json_object))

        with open(f"{self.experiment_dir}/loss_log.txt", "w") as file1:
            file1.write(loss_log_title)
            file1.write("\n")

        mkdir(f"{self.experiment_dir}/checkpoints")
        mkdir(f"{self.experiment_dir}/tb")
        mkdir(f"{self.experiment_dir}/visuals")
