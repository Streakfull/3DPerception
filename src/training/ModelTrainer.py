from typing import Type

import numpy as np
import torch
import yaml
from cprint import *
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataSet
from src.datasets.shape_net.shape_net_vox import ShapeNetVox
from src.models.dummy_classifier import DummyClassifier
from src.training.DataLoaderHandler import DataLoaderHandler
from src.training.Logger import Logger, TrainingVariables


class ModelTrainer:
    def __init__(self, dataset_type: Type[BaseDataSet] = ShapeNetVox, configs_path="./configs/global_configs.yaml",
                 test_size=0.2):
        self.dataset_type = dataset_type
        with open(configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
        self.training_config = self.global_configs["training"]
        self.device = self.training_config["device"]
        self.experiment_dir = Logger(self.training_config).experiment_dir

        self.train_dataloader, self.validation_dataloader = (
            DataLoaderHandler(global_configs=self.global_configs,
                              dataset_type=dataset_type,
                              batch_size=self.training_config['batch_size'],
                              test_size=self.training_config['test_size'],
                              num_workers=self.training_config['num_workers']).get_dataloaders())
        self.model, self.train_vars = self._prepare_model()

    def _prepare_model(self):
        model_configs = self.global_configs["model"]["dummy_classifier"]
        if "cpu" == self.device:
            cprint.warn('Using CPU')
        else:
            cprint.ok('Using device:', self.device)

        model = DummyClassifier(model_configs)
        model.to(self.device)

        if self.training_config["load_ckpt"]:
            model.load_ckpt(self.training_config['ckpt_path'])

        if torch.cuda.is_available():
            torch.cuda.mem_get_info()

        train_vars = TrainingVariables(experiment_dir=self.experiment_dir, train_loss_running=0., best_loss_val=np.inf,
                                       start_iteration=self.training_config["start_iteration"], last_loss=0.)
        return model, train_vars

    def _train_one_epoch(self, epoch, writer):
        train_loss_running = 0.
        iteration = 0
        with open(self.train_vars.loss_log_name, "a") as log_file:
            log_file.write(f'** Epoch: {epoch} **\n')
        for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            iteration += 1
            if iteration <= self.train_vars.start_iteration:
                continue
            self.dataset_type.move_batch_to_device(batch, self.device)
            self.model.step(batch)
            metrics = self.model.get_metrics()
            loss = metrics["loss"]
            train_loss_running += loss
            # log loss
            if iteration % self.training_config["append_loss_every"] == (
                    self.training_config["append_loss_every"] - 1) or (
                    epoch == 0 and iteration == 0):
                message = '(epoch: %d, iters: %d, loss: %.6f)' % (epoch, iteration, loss.item())
                with open(self.train_vars.loss_log_name, "a") as log_file:
                    log_file.write('%s\n' % message)

            # visualization step
            # if iteration % self.training_config["visualize_every"] == (self.training_config["visualize_every"] - 1):
            #     # Do visualizations here
            #     pass

            # log writer
            if iteration % self.training_config['print_every'] == (self.training_config['print_every'] - 1) or (
                    epoch == 0 and iteration == 0):
                avg_train_loss = train_loss_running / iteration
                cprint.warn(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {avg_train_loss:.6f}')
                writer.add_scalar("Train/Loss", avg_train_loss, iteration)
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
                loss_val = 0.
                index_batch = 0
                for _, batch_val in tqdm(enumerate(self.validation_dataloader), total=len(
                        self.validation_dataloader)):
                    with torch.no_grad():
                        self.dataset_type.move_batch_to_device(batch_val, self.device)
                        self.model.inference(batch_val)
                        metrics = self.model.get_metrics()
                        loss_val += metrics["loss"]
                        index_batch += 1
                avg_loss_val = loss_val / index_batch

                # Do visualizations here
                if avg_loss_val < self.train_vars.best_loss_val:
                    self.model.save(self.train_vars.model_checkpoint_path, "best")
                    self.train_vars.best_loss_val = avg_loss_val

                cprint.warn(
                    f'[{epoch:03d}/{batch_idx:05d}] val_loss: {avg_loss_val:.6f} | best_loss_val: '
                    f'{self.train_vars.best_loss_val:.6f}')
                writer.add_scalar("Validation/Loss", avg_loss_val, iteration)
                writer.add_scalars('Validation/LossComparison',
                                   {'Training': self.train_vars.last_loss, 'Validation': avg_loss_val},
                                   iteration)
                writer.flush()

    def train(self):
        start_epoch = self.training_config["start_epoch"]
        for epoch in tqdm(range(self.training_config['n_epochs'])):
            if epoch < start_epoch:
                continue
            self._train_one_epoch(epoch, self.train_vars.writer)
            if epoch % self.training_config["save_every_nepochs"] == 0:
                self.model.save(self.train_vars.model_checkpoint_path, epoch)
            self.model.update_lr()
            self.train_vars.writer.close()
        self.model.save(self.train_vars.model_checkpoint_path)
