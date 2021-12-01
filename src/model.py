#import
import argparse
from pytorch_lightning import LightningModule
import sys
from os.path import dirname, basename, isfile, join
import torch.nn as nn
import torch
import timm
from glob import glob


#class
class BaseModel(LightningModule):
    def __init__(self, loss_function, checkpoint_path) -> None:
        super().__init__()
        self.loss_function = loss_function
        self.checkpoint_path = checkpoint_path

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

    def load_from_checkpoint(self, model, device):
        device = device if device == 'cuda' and torch.cuda.is_available(
        ) else 'cpu'
        map_location = torch.device(device=device)
        checkpoint = torch.load(f=self.checkpoint_path,
                                map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])
        return model


#TODO: parse_optimizers,parse_lr_schedulers,load_from_checkpoint.
class SupervisedModel(BaseModel):
    def __init__(self, loss_function, checkpoint_path, model_name, in_chans,
                 classes, data_balance, root) -> None:
        super().__init__(loss_function=loss_function,
                         checkpoint_path=checkpoint_path)
        self.backbone_model = self.create_backbone_model(model_name=model_name,
                                                         in_chans=in_chans,
                                                         classes=classes)
        self.activation_function = nn.Sigmoid()
        self.loss_function = self.create_loss_function(
            data_balance=data_balance, root=root, classes=classes)

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


if __name__ == '__main__':
    #parameters
    project_parameters = {
        'loss_function': 'BCEWithLogitsLoss',
        'checkpoint_path': None,
        'device': 'cpu',
        'model_name': 'tf_mobilenetv3_small_minimal_100',
        'in_chans': 3,
        'classes':
        ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'],
        'data_balance': True,
        'root': './data/Intel_Image_Classification'
    }
    project_parameters = argparse.Namespace(**project_parameters)

    #
    model = SupervisedModel(loss_function=project_parameters.loss_function,
                            checkpoint_path=project_parameters.checkpoint_path,
                            model_name=project_parameters.model_name,
                            in_chans=project_parameters.in_chans,
                            classes=project_parameters.classes,
                            data_balance=project_parameters.data_balance,
                            root=project_parameters.root)
