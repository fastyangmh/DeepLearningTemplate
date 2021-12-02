#import
import argparse
from pytorch_lightning import LightningModule
import sys
from os.path import dirname, basename, isfile, join
import torch.nn as nn
import torch
import timm
from glob import glob
import torch.optim as optim
from torchmetrics import Accuracy, ConfusionMatrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#def
def load_from_checkpoint(device, checkpoint_path, model):
    device = device if device == 'cuda' and torch.cuda.is_available(
    ) else 'cpu'
    map_location = torch.device(device=device)
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def load_from_checkpoint_for_supervised_model(
    device,
    checkpoint_path,
    classes,
    model,
):
    device = device if device == 'cuda' and torch.cuda.is_available(
    ) else 'cpu'
    map_location = torch.device(device=device)
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    # change the number of output
    for key in checkpoint['state_dict'].keys():
        if 'classifier.bias' in key or 'classifier.weight' in key:
            if checkpoint['state_dict'][key].shape[0] != len(classes):
                temp = checkpoint['state_dict'][key]
                checkpoint['state_dict'][key] = torch.stack([temp.mean(0)] *
                                                            len(classes), 0)
    # change the weight of loss function
    if model.loss_function.weight is None:
        if 'loss_function.weight' in checkpoint['state_dict']:
            # delete loss_function.weight in the checkpoint
            del checkpoint['state_dict']['loss_function.weight']
    else:
        # override loss_function.weight with model.loss_function.weight
        checkpoint['state_dict'][
            'loss_function.weight'] = model.loss_function.weight
    model.load_state_dict(checkpoint['state_dict'])
    return model


#class
class BaseModel(LightningModule):
    def __init__(self, loss_function, optimizers_config, lr,
                 lr_schedulers_config) -> None:
        super().__init__()
        self.loss_function = loss_function
        self.optimizers_config = optimizers_config
        self.lr = lr
        self.lr_schedulers_config = lr_schedulers_config

    def import_class_from_file(self, filepath):
        sys.path.append('{}'.format(dirname(filepath)))
        filename = basename(filepath)[:-3]
        # assume the class name in file is SelfDefinedModel
        class_name = 'SelfDefinedModel'
        # check if the class_name exists in the file
        exec('import {}'.format(filename))
        classes_in_file = dir(eval('{}'.format(filename)))
        assert class_name in classes_in_file, 'please check the self defined model.\nthe {} does not exist the {} class.'.format(
            filepath, class_name)
        exec('from {} import {}'.format(filename, class_name))
        return eval(class_name)

    def create_loss_function(self):
        assert self.loss_function in dir(
            nn
        ), 'please check the loss_function argument.\nloss_function: {}\nvalid: {}'.format(
            self.loss_function, [v for v in dir(nn) if v[0].isupper()])
        return eval('nn.{}()'.format(self.loss_function))

    def parse_optimizers(self, params):
        optimizers = []
        for optimizer_name in self.optimizers_config.keys():
            if optimizer_name in dir(optim):
                if self.optimizers_config[optimizer_name] is None:
                    # create an optimizer with default values
                    optimizers.append(
                        eval('optim.{}(params=params, lr={})'.format(
                            optimizer_name, self.lr)))
                else:
                    # create an optimizer using the values given by the user
                    optimizer_arguments = [
                        '{}={}'.format(a, b) for a, b in
                        self.optimizers_config[optimizer_name].items()
                    ]
                    optimizer_arguments = ','.join(optimizer_arguments)
                    optimizers.append(
                        eval('optim.{}(params=params, lr={}, {})'.format(
                            optimizer_name, self.lr, optimizer_arguments)))
            else:
                assert False, 'please check the optimizer name in the optimizers_config argument.\noptimizer name: {}\nvalid: {}'.format(
                    optimizer_name, [v for v in dir(optim) if v[0].isupper()])
        return optimizers

    def parse_lr_schedulers(self, optimizers):
        lr_schedulers = []
        for idx, lr_scheduler_name in enumerate(
                self.lr_schedulers_config.keys()):
            if lr_scheduler_name in dir(optim.lr_scheduler):
                if self.lr_schedulers_config[lr_scheduler_name] is None:
                    # create a learning rate scheduler with default values
                    lr_schedulers.append(
                        eval(
                            'optim.lr_scheduler.{}(optimizer=optimizers[idx])'.
                            format(lr_scheduler_name)))
                else:
                    # create a learning rate scheduler using the values given by the user
                    lr_schedulers_arguments = [
                        '{}={}'.format(a, b) for a, b in
                        self.lr_schedulers_config[lr_scheduler_name].items()
                    ]
                    lr_schedulers_arguments = ','.join(lr_schedulers_arguments)
                    lr_schedulers.append(
                        eval(
                            'optim.lr_scheduler.{}(optimizer=optimizers[idx], {})'
                            .format(lr_scheduler_name,
                                    lr_schedulers_arguments)))
            else:
                assert False, 'please check the learning scheduler name in the lr_schedulers_config argument.\nlearning scheduler name: {}\nvalid: {}'.format(
                    lr_scheduler_name,
                    [v for v in dir(optim) if v[0].isupper()])
        return lr_schedulers


class SupervisedModel(BaseModel):
    def __init__(self, loss_function, optimizers_config, lr,
                 lr_schedulers_config, model_name, in_chans, classes,
                 data_balance, root) -> None:
        super().__init__(loss_function=loss_function,
                         optimizers_config=optimizers_config,
                         lr=lr,
                         lr_schedulers_config=lr_schedulers_config)
        self.backbone_model = self.create_backbone_model(model_name=model_name,
                                                         in_chans=in_chans,
                                                         classes=classes)
        self.activation_function = nn.Sigmoid()
        self.loss_function = self.create_loss_function(
            data_balance=data_balance, root=root, classes=classes)
        self.accuracy_function = Accuracy()
        self.confusion_matrix_function = ConfusionMatrix(
            num_classes=len(classes))
        self.classes = classes
        self.stage_index = 0

    def create_backbone_model(self, model_name, in_chans, classes):
        if model_name in timm.list_models():
            backbone_model = timm.create_model(model_name=model_name,
                                               pretrained=True,
                                               in_chans=in_chans,
                                               num_classes=len(classes))
        elif isfile(model_name):
            class_name = self.import_class_from_file(filepath=model_name)
            backbone_model = class_name(in_chans=in_chans,
                                        num_classes=len(classes))
        else:
            assert False, 'please check the model_name argument.\nthe model_name value is {}.'.format(
                model_name)
        return backbone_model

    def get_files(self, filepath, extensions):
        files = []
        for v in extensions:
            files += glob(join(filepath, '*{}'.format(v)))
        return files

    def calculate_weight(self, root, classes):
        weight = {}
        for c in classes:
            files = self.get_files(filepath=join(root, 'train/{}'.format(c)),
                                   extensions=('.jpg', '.jpeg', '.png', '.ppm',
                                               '.bmp', '.pgm', '.tif', '.tiff',
                                               '.webp'))
            weight[c] = len(files)
        weight = {
            c: 1 - (weight[c] / sum(weight.values()))
            for c in weight.keys()
        }
        return weight

    def create_loss_function(self, data_balance, root, classes):
        loss_function = super().create_loss_function()
        if data_balance:
            weight = self.calculate_weight(root=root, classes=classes)
            weight = torch.Tensor(list(weight.values()))
        else:
            weight = None
        loss_function.weight = weight
        return loss_function

    def configure_optimizers(self):
        optimizers = self.parse_optimizers(params=self.parameters())
        if self.lr_schedulers_config is not None:
            lr_schedulers = self.parse_lr_schedulers(optimizers=optimizers)
            return optimizers, lr_schedulers
        else:
            return optimizers

    def forward(self, x):
        return self.activation_function(self.backbone_model(x))

    def shared_step(self, batch):
        x, y = batch
        y_hat = self.backbone_model(x)
        loss = self.loss_function(y_hat, y)
        accuracy = self.accuracy_function(self.activation_function(y_hat),
                                          y.argmax(-1))
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch=batch)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('train_accuracy',
                 accuracy,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch=batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone_model(x)
        loss = self.loss_function(y_hat, y)
        accuracy = self.accuracy_function(self.activation_function(y_hat),
                                          y.argmax(-1))
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        # NOTE: in the 1.4.9+ version of PyTorch Lightning,
        # if deterministic is set to True,
        # an error will occur while calculating the confusion matrix.
        # it can use torch.use_deterministic_algorithms(False) to solved.
        confusion_matrix_step = self.confusion_matrix_function(
            y_hat.argmax(-1), y.argmax(-1)).cpu().data.numpy()
        loss_step = loss.item()
        accuracy_step = accuracy.item()
        return {
            'confusion_matrix': confusion_matrix_step,
            'loss': loss_step,
            'accuracy': accuracy_step
        }

    def test_epoch_end(self, test_outs):
        stages = ['train', 'val', 'test']
        print('\ntest the {} dataset'.format(stages[self.stage_index]))
        print('the {} dataset confusion matrix:'.format(
            stages[self.stage_index]))
        confusion_matrix = np.sum([v['confusion_matrix'] for v in test_outs],
                                  0)
        loss = np.mean([v['loss'] for v in test_outs])
        accuracy = np.mean([v['accuracy'] for v in test_outs])
        confusion_matrix = pd.DataFrame(data=confusion_matrix,
                                        columns=self.classes,
                                        index=self.classes).astype(int)
        print(confusion_matrix)
        plt.figure(figsize=[11.2, 6.3])
        plt.title('{}\nloss: {}\naccuracy: {}'.format(stages[self.stage_index],
                                                      loss, accuracy))
        figure = sns.heatmap(data=confusion_matrix,
                             cmap='Spectral',
                             annot=True,
                             fmt='g').get_figure()
        plt.yticks(rotation=0)
        plt.ylabel(ylabel='Actual class')
        plt.xlabel(xlabel='Predicted class')
        plt.close()
        self.logger.experiment.add_figure(
            '{} confusion matrix'.format(stages[self.stage_index]), figure,
            self.current_epoch)
        self.stage_index += 1


#TODO: add unsupervised model.

if __name__ == '__main__':
    #parameters
    project_parameters = {
        'loss_function': 'BCEWithLogitsLoss',
        'checkpoint_path': None,
        'optimizers_config': {
            'Adam': {
                'betas': [0.9, 0.999],
                'eps': 1e-08,
                'weight_decay': 0,
                'amsgrad': False
            }
        },
        'lr': 1e-3,
        'lr_schedulers_config': {
            'CosineAnnealingLR': {
                'T_max': 10
            }
        },
        'model_name': 'tf_mobilenetv3_small_minimal_100',
        'in_chans': 3,
        'classes':
        ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'],
        'data_balance': True,
        'root': './data/Intel_Image_Classification',
        'device': 'cpu'
    }
    project_parameters = argparse.Namespace(**project_parameters)

    #
    model = SupervisedModel(
        loss_function=project_parameters.loss_function,
        optimizers_config=project_parameters.optimizers_config,
        lr=project_parameters.lr,
        lr_schedulers_config=project_parameters.lr_schedulers_config,
        model_name=project_parameters.model_name,
        in_chans=project_parameters.in_chans,
        classes=project_parameters.classes,
        data_balance=project_parameters.data_balance,
        root=project_parameters.root)