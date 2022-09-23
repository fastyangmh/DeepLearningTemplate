# import
from typing import Any, Iterable, List, Dict
from pytorch_lightning import LightningModule
import argparse
import sys
from os.path import dirname, basename
import torch.nn as nn
import torch.optim as optim
import torch


# class
class BaseModel(LightningModule):
    def __init__(self, project_parameters: argparse.Namespace, *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimizers_config = project_parameters.optimizers_config
        self.lr = project_parameters.lr
        self.lr_schedulers_config = project_parameters.lr_schedulers_config
        self.model_mode = project_parameters.model_mode
        self.save_hyperparameters()

    def import_class_from_file(self, filepath: str, class_name: str):
        sys.path.append(f'{dirname(filepath)}')
        module_name = basename(filepath)[:-3]
        exec(f'import {module_name}')
        classes_in_file = dir(eval(f'{module_name}'))
        assert class_name in classes_in_file, f'please check the self defined model.\nthe {filepath} does not exist the {class_name} class.'
        exec(f'from {module_name} import {class_name}')
        return eval(class_name)

    def create_loss_function(self, loss_function_name: str):
        valid_loss_function_name = [v for v in dir(nn) if 'Loss' in v]
        assert loss_function_name in valid_loss_function_name, f'please check the loss_function_name argument.\nloss_function: {loss_function_name}\nvalid: {valid_loss_function_name}'
        return eval(f'nn.{loss_function_name}()')

    def parse_optimizers(self, params: Iterable):
        valid_optimizer_name = [v for v in dir(optim) if v[0].isupper()]
        optimizers = []
        for optimizer_name in self.optimizers_config.keys():
            if optimizer_name in valid_optimizer_name:
                if self.optimizers_config[optimizer_name] is None:
                    # create an optimizer with default values
                    optimizers.append(
                        eval(
                            f'optim.{optimizer_name}(params=params, lr={self.lr})'
                        ))
                else:
                    # create an optimizer using the values given by the user
                    optimizer_arguments = [
                        f'{a}={b}' for a, b in
                        self.optimizers_config[optimizer_name].items()
                    ]
                    optimizer_arguments = ','.join(optimizer_arguments)
                    optimizers.append(
                        eval(
                            f'optim.{optimizer_name}(params=params, lr={self.lr}, {optimizer_arguments})'
                        ))
            else:
                assert False, f'please check the optimizer name in the optimizers_config argument.\noptimizer name: {optimizer_name}\nvalid: {valid_optimizer_name}'
        return optimizers

    def parse_lr_schedulers(self, optimizers: List):
        valid_lr_scheduler_name = [
            v for v in dir(optim.lr_scheduler) if v[0].isupper()
        ]
        lr_schedulers = []
        for idx, lr_scheduler_name in enumerate(
                self.lr_schedulers_config.keys()):
            if lr_scheduler_name in valid_lr_scheduler_name:
                if self.lr_schedulers_config[lr_scheduler_name] is None:
                    # create a learning rate scheduler with default values
                    lr_schedulers.append(
                        eval(
                            f'optim.lr_scheduler.{lr_scheduler_name}(optimizer=optimizers[idx])'
                        ))
                else:
                    # create a learning rate scheduler using the values given by the user
                    lr_schedulers_arguments = [
                        f'{a}={b}' for a, b in
                        self.lr_schedulers_config[lr_scheduler_name].items()
                    ]
                    lr_schedulers_arguments = ','.join(lr_schedulers_arguments)
                    lr_schedulers.append(
                        eval(
                            f'optim.lr_scheduler.{lr_scheduler_name}(optimizer=optimizers[idx], {lr_schedulers_arguments})'
                        ))
            else:
                assert False, f'please check the learning scheduler name in the lr_schedulers_config argument.\nlearning scheduler name: {lr_scheduler_name}\nvalid: {valid_lr_scheduler_name}'
        return lr_schedulers

    def configure_optimizers(self):
        optimizers = self.parse_optimizers(params=self.parameters())
        if self.lr_schedulers_config is not None:
            lr_schedulers = self.parse_lr_schedulers(optimizers=optimizers)
            return optimizers, lr_schedulers
        else:
            return optimizers

    def check_checkpoint(self, checkpoint: Dict[str, Any]):
        #modify the number of model outputs in pretrained weight
        for key in checkpoint['state_dict'].keys():
            if 'classifier.bias' in key or 'classifier.weight' in key:
                if checkpoint['state_dict'][key].shape[0] != len(self.classes):
                    temp = checkpoint['state_dict'][key]
                    checkpoint['state_dict'][key] = torch.stack(
                        [temp.mean(0)] * len(self.classes), 0)

        # change the weight of loss function
        if self.loss_function.weight is None:
            if 'loss_function.weight' in checkpoint['state_dict']:
                # delete loss_function.weight in the checkpoint
                del checkpoint['state_dict']['loss_function.weight']
        else:
            # override loss_function.weight with model.loss_function.weight
            checkpoint['state_dict'][
                'loss_function.weight'] = self.loss_function.weight
        return checkpoint

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.model_mode == 'supervised':
            checkpoint = self.check_checkpoint(checkpoint=checkpoint)
        return super().on_load_checkpoint(checkpoint)
