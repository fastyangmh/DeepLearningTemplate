# import
from typing import Any, Iterable, List, Dict, Tuple, Union
from pytorch_lightning import LightningModule
import argparse
import sys
from os.path import dirname, basename, isfile, join, splitext
import torch.nn as nn
import torch.optim as optim
import torch
import timm
from glob import glob
from torchmetrics import Accuracy, ConfusionMatrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
try:
    from . import IMG_EXTENSIONS, AUDIO_EXTENSIONS, SERIES_EXTENSIONS
except:
    from dataset import IMG_EXTENSIONS, AUDIO_EXTENSIONS, SERIES_EXTENSIONS


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

    def parse_loss_function(self, loss_function_name: str):
        valid_loss_function_name = [v for v in dir(nn) if 'Loss' in v]
        assert loss_function_name in valid_loss_function_name, f'please check the loss_function_name argument.\nloss_function_name: {loss_function_name}\nvalid: {valid_loss_function_name}'
        return eval(f'nn.{loss_function_name}()')

    def parse_last_activation_function(self,
                                       last_activation_function_name: str):
        valid_last_activation_function_name = ['Softmax', 'Sigmoid']
        assert last_activation_function_name in valid_last_activation_function_name, f'please check the last_activation_function_name argument.\nactivation_function_name: {last_activation_function_name}\nvalid: {valid_last_activation_function_name}'
        return eval(f'nn.{last_activation_function_name}()')

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


class SupervisedModel(BaseModel):
    def __init__(self, project_parameters: argparse.Namespace, *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(project_parameters, *args, **kwargs)
        self.backbone_model = self.parse_backbone_model(
            model_name=project_parameters.model_name,
            in_chans=project_parameters.in_chans,
            classes=project_parameters.classes)
        self.last_activation_function = self.parse_last_activation_function(
            last_activation_function_name=project_parameters.
            last_activation_function_name)
        if project_parameters.file_extensions in [
                'IMG_EXTENSIONS', 'AUDIO_EXTENSIONS', 'SERIES_EXTENSIONS'
        ]:
            self.extensions = eval(project_parameters.file_extensions)
        else:
            self.extensions = project_parameters.file_extensions
        self.loss_function = self.parse_loss_function(
            loss_function_name=project_parameters.loss_function_name,
            weighted_loss=project_parameters.weighted_loss,
            root=project_parameters.root,
            classes=project_parameters.classes)
        self.accuracy_function = Accuracy()
        self.confusion_matrix_function = ConfusionMatrix(
            num_classes=len(project_parameters.classes))
        self.classes = project_parameters.classes
        self.stage_index = 0

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
        checkpoint = self.check_checkpoint(checkpoint=checkpoint)
        return super().on_load_checkpoint(checkpoint)

    def parse_backbone_model(self, model_name: Union[str, List], in_chans: int,
                             classes: List):
        if isinstance(model_name, list):
            model_name, class_name = model_name
        if model_name in timm.list_models():
            backbone_model = timm.create_model(model_name=model_name,
                                               pretrained=True,
                                               in_chans=in_chans,
                                               num_classes=len(classes))
        elif isfile(model_name):
            model_cls = self.import_class_from_file(filepath=model_name,
                                                    class_name=class_name)
            backbone_model = model_cls(in_chans=in_chans,
                                       num_classes=len(classes))
        else:
            assert False, f'please check the model_name argument.\nthe model_name value is {model_name}'
        return backbone_model

    def calculate_weight(self, root: str, classes: List):
        counts = {}
        for cls in classes:
            files = [
                f for f in glob(join(root, f'train/{cls}/*'))
                if splitext(f)[-1] in self.extensions
            ]
            counts[cls] = len(files)
        weight = {
            cls: min(counts.values()) / counts[cls]
            for cls in counts.keys()
        }
        return weight

    def parse_loss_function(self, loss_function_name: str, weighted_loss: bool,
                            root: str, classes: List):
        loss_function = super().parse_loss_function(loss_function_name)
        if weighted_loss:
            weight = self.calculate_weight(root=root, classes=classes)
            weight = torch.Tensor(list(weight.values()))
        else:
            weight = None
        loss_function.weight = weight
        return loss_function

    def plot_confusion_matrix(self, title: str, confusion_matrix: pd.DataFrame,
                              stage: str):
        #plt.figure(figsize=[16, 9])  #add this line if the confusion matrix figure is too small
        plt.title(title)
        figure = sns.heatmap(data=confusion_matrix,
                             cmap='Blues',
                             annot=True,
                             fmt='g').get_figure()
        plt.yticks(rotation=0)
        plt.ylabel(ylabel='Actual class')
        plt.xlabel(xlabel='Predicted class')
        plt.tight_layout()
        plt.close()
        self.logger.experiment.add_figure(f'{stage} confusion matrix', figure,
                                          self.current_epoch)

    def forward(self, x: torch.Tensor):
        return self.last_activation_function(self.backbone_model(x))

    def shared_step(self, batch: Tuple, stage: str):
        x, y = batch
        y_hat = self.backbone_model(x)
        loss = self.loss_function(y_hat, y)
        accuracy = self.accuracy_function(self.last_activation_function(y_hat),
                                          y.argmax(-1))
        if stage == 'test':
            return loss, accuracy, y, y_hat
        return loss, accuracy

    def training_step(self, batch: Tuple, batch_idx: int):
        loss, accuracy = self.shared_step(batch=batch, stage='train')
        self.log(name='train_loss',
                 value=loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log(name='train_accuracy',
                 value=accuracy,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int):
        loss, accuracy = self.shared_step(batch=batch, stage='val')
        self.log(name='val_loss', value=loss, prog_bar=True)
        self.log(name='val_accuracy', value=accuracy, prog_bar=True)

    def test_step(self, batch: Tuple, batch_idx: int):
        loss, accuracy, y, y_hat = self.shared_step(batch=batch, stage='test')
        self.log(name='test_loss', value=loss)
        self.log(name='test_accuracy', value=accuracy)
        self.confusion_matrix_function.update(
            preds=self.last_activation_function(y_hat).argmax(-1),
            target=y.argmax(-1))
        loss_step = loss.item()
        accuracy_step = accuracy.item()
        return {'loss': loss_step, 'accuracy': accuracy_step}

    def test_epoch_end(self, outputs: Dict):
        stages = ['train', 'val', 'test']
        confusion_matrix = self.confusion_matrix_function.compute().cpu(
        ).data.numpy()
        confusion_matrix = pd.DataFrame(data=confusion_matrix,
                                        columns=self.classes,
                                        index=self.classes).astype(int)
        print(f'\ntesting the {stages[self.stage_index]} dataset')
        print(
            f'the {stages[self.stage_index]} dataset confusion matrix:\n{confusion_matrix}'
        )
        loss = np.mean([v['loss'] for v in outputs])
        accuracy = np.mean([v['accuracy'] for v in outputs])
        title = f'{stages[self.stage_index]}\nloss: {loss}\naccuracy: {accuracy}'
        stage = stages[self.stage_index]
        self.plot_confusion_matrix(title=title,
                                   confusion_matrix=confusion_matrix,
                                   stage=stage)
        self.stage_index += 1
        self.confusion_matrix_function.reset()


#global parameters
VALID_MODEL = [
    name
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if 'Model' in name and 'Base' not in name
]


#def
def create_model(project_parameters: argparse.Namespace):
    if project_parameters.weighted_loss and project_parameters.mode not in [
            'train', 'tuning'
    ]:
        print(
            'please check the weighted_loss and mode arguments.\nyou set weighted_loss to True,\nbut these arguments are only valid in train and tuning mode.'
        )
        project_parameters.weighted_loss = False
    assert project_parameters.model_mode in VALID_MODEL, f'please check the model_mode argument.\nmodel_mode: {project_parameters.model_mode}\nvalid: {VALID_MODEL}'
    if project_parameters.checkpoint_path is not None:
        return eval(
            f'{project_parameters.model_mode}.load_from_checkpoint(checkpoint_path=project_parameters.checkpoint_path, project_parameters=project_parameters)'
        )
    else:
        return eval(
            f'{project_parameters.model_mode}(project_parameters=project_parameters)'
        )