import numpy as np
import torch
import yaml
from cprint import *
from tqdm import tqdm

from src.training.DataLoaderHandler import DataLoaderHandler
from src.training.Logger import Logger
from src.training.ModelBuilder import ModelBuilder
from src.training.TrainingVariables import TrainingVariables


class ModelTrainer:
    def __init__(self, configs_path="./configs/global_configs.yaml"):
        with open(configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
        self.training_config = self.global_configs["training"]
        self.device = self.training_config["device"]
        self.logger = Logger(self.training_config)
        self.experiment_dir = self.logger.experiment_dir
        data_loader_handler = DataLoaderHandler(global_configs=self.global_configs,
                                                batch_size=self.training_config['batch_size'],
                                                test_size=self.training_config['test_size'],
                                                num_workers=self.training_config['num_workers'])
        self.train_dataloader, self.validation_dataloader = data_loader_handler.get_dataloaders()
        self.dataset_type = data_loader_handler.dataset_type
        self.train_vars = TrainingVariables(experiment_dir=self.experiment_dir, train_loss_running=0.,
                                            best_loss_val=np.inf,
                                            start_iteration=self.training_config["start_iteration"], last_loss=0.)
        self.model = ModelBuilder(self.global_configs["model"], self.training_config).get_model()

    def _train_one_epoch(self, epoch):
        train_loss_running = 0.
        iteration = 0
        self.logger.start_new_epoch(epoch)
        for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            iteration += 1
            if iteration <= self.train_vars.start_iteration:
                continue
            self.dataset_type.move_batch_to_device(batch, self.device)
            self.model.step(batch)
            loss = self.model.get_metrics().get("loss")
            train_loss_running += loss

            # log loss
            self.logger.log_loss(epoch, iteration, loss)

            # visualization step
            # if iteration % self.training_config["visualize_every"] == (self.training_config["visualize_every"] - 1):
            #     # Do visualizations here
            #     pass

            # log writer
            if iteration % self.training_config['print_every'] == (self.training_config['print_every'] - 1) or (
                    epoch == 0 and iteration == 0):
                avg_train_loss = train_loss_running / iteration
                cprint.warn(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {avg_train_loss:.6f}')
                self.logger.add_scalar("Train/Loss", avg_train_loss, iteration)
                self.train_vars.last_loss = avg_train_loss
                train_loss_running = 0.

            # saving step
            if iteration % self.training_config['save_every'] == (self.training_config['save_every'] - 1):
                self.model.save(self.train_vars.model_checkpoint_path, "latest")
                pass

            # validation step
            if iteration % self.training_config['validate_every'] == (self.training_config['validate_every'] - 1) or (
                    epoch == 0 and iteration == 0):
                cprint.ok("Running Validation")
                self.model.eval()
                val_loss_running = 0.
                index_batch = 0
                for _, batch_val in tqdm(enumerate(self.validation_dataloader), total=len(
                        self.validation_dataloader)):
                    with torch.no_grad():
                        self.dataset_type.move_batch_to_device(batch_val, self.device)
                        self.model.inference(batch_val)
                        val_loss = self.model.get_metrics().get("loss")
                        val_loss_running += val_loss
                        index_batch += 1
                avg_loss_val = val_loss_running / index_batch

                # Do visualizations here
                if avg_loss_val < self.train_vars.best_loss_val:
                    self.model.save(self.train_vars.model_checkpoint_path, "best")
                    self.train_vars.best_loss_val = avg_loss_val

                cprint.warn(
                    f'[{epoch:03d}/{batch_idx:05d}] val_loss: {avg_loss_val:.6f} | best_loss_val: '
                    f'{self.train_vars.best_loss_val:.6f}')
                self.logger.add_scalar("Validation/Loss", avg_loss_val, iteration)
                self.logger.add_scalars('Validation/LossComparison',
                                        {'Training': self.train_vars.last_loss, 'Validation': avg_loss_val},
                                        iteration)
                self.logger.flush_writer()

    def train(self):
        start_epoch = self.training_config["start_epoch"]
        for epoch in tqdm(range(self.training_config['n_epochs'])):
            if epoch < start_epoch:
                continue
            self._train_one_epoch(epoch)
            if epoch % self.training_config["save_every_nepochs"] == 0:
                self.model.save(self.train_vars.model_checkpoint_path, epoch)
            self.model.update_lr()
            self.logger.close_writer()
        self.model.save(self.train_vars.model_checkpoint_path)
