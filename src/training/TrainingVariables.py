from dataclasses import dataclass

import numpy as np
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainingVariables:
    def __init__(self, experiment_dir, train_loss_running=0., best_loss_val=np.inf, start_iteration=0, last_loss=0.):
        self.train_loss_running = train_loss_running
        self.best_loss_val = best_loss_val
        self.start_iteration = start_iteration
        tb_dir = f"{experiment_dir}/tb"
        self.writer = SummaryWriter(log_dir=tb_dir)
        self.model_checkpoint_path = f"{experiment_dir}/checkpoints"
        self.visuals_path = f"{experiment_dir}/visuals"
        self.last_loss = last_loss
