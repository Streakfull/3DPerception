import json
import time
from datetime import datetime
from pathlib import Path

from cprint import *

from src.utils.util import mkdir


class Logger:
    def __init__(self, training_config):
        cprint.ok(training_config)
        logs_dir = training_config["logs_dir"]
        self.append_loss_every = training_config["append_loss_every"]

        self.experiment_dir = Path(logs_dir) / training_config['name'] / datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S")
        mkdir(self.experiment_dir)
        self.loss_log_file_name = f"{self.experiment_dir}/loss_log.txt"

        self.make_log_files(training_config)
        self.make_dirs()

    def make_log_files(self, training_config):
        with open(f"{self.experiment_dir}/description.txt", "w") as file1:
            description = training_config["description"]
            file1.write(description)

        with open(f"{self.experiment_dir}/global_configs.json", "w") as file1:
            json_object = json.dumps(training_config, indent=4)
            file1.write(str(json_object))

        with open(self.loss_log_file_name, "w") as file1:
            loss_log_title = "Loss Log " + time.strftime("%Y-%m-%d")
            file1.write(loss_log_title)
            file1.write("\n")

    def make_dirs(self):
        mkdir(f"{self.experiment_dir}/checkpoints")
        mkdir(f"{self.experiment_dir}/tb")
        mkdir(f"{self.experiment_dir}/visuals")

    def start_new_epoch(self, epoch):
        with open(self.loss_log_file_name, "a") as log_file:
            log_file.write(f'** Epoch: {epoch} **\n')

    def log_loss(self, epoch, iteration, loss):
        if iteration % self.append_loss_every == (self.append_loss_every - 1) or (epoch == 0 and iteration == 0):
            message = '(epoch: %d, iters: %d, loss: %.6f)' % (epoch, iteration, loss.item())
            with open(self.loss_log_file_name, "a") as log_file:
                log_file.write('%s\n' % message)
