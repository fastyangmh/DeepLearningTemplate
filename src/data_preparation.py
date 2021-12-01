#import
import argparse
from typing import Any, Optional, Callable, Union, Tuple, List, Dict
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, DatasetFolder
import random
from torchaudio.datasets import SPEECHCOMMANDS
from pathlib import Path
from torch import Tensor
import os
from collections import defaultdict
import torchvision
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import lowpass_biquad, highpass_biquad
from PIL import Image
from pytorch_lightning import LightningDataModule
import torch
from os.path import join
from torch.utils.data import random_split, DataLoader
from os import makedirs


#def
def parse_transforms(transforms_config):
    if transforms_config is None:
        return {'train': None, 'val': None, 'test': None, 'predict': None}
    transforms_dict = defaultdict(list)
    for stage in transforms_config.keys():
        if transforms_config[stage] is None:
            transforms_dict[stage] = None
            continue
        for name, value in transforms_config[stage].items():
            if name in ['DigitalFilter', 'PadWaveform']:
                name = name
            elif name in dir(torchvision.transforms):
                name = 'torchvision.transforms.{}'.format(name)
            elif name in dir(torchaudio.transforms):
                name = 'torchaudio.transforms.{}'.format(name)
            if value is None:
                transforms_dict[stage].append(eval('{}()'.format(name)))
            else:
                if type(value) is dict:
                    transform_arguments = []
                    for a, b in value.items():
                        if type(b) is str:
                            arg = '{}="{}"'.format(a, b)
                        else:
                            arg = '{}={}'.format(a, b)
                        transform_arguments.append(arg)
                    transform_arguments = ','.join(transform_arguments)
                    value = transform_arguments
                transforms_dict[stage].append(
                    eval('{}({})'.format(name, value)))
        transforms_dict[stage] = torchvision.transforms.Compose(
            transforms_dict[stage])
    return transforms_dict


def parse_target_transforms(target_transforms_config, classes):
    if target_transforms_config is None:
        return {'train': None, 'val': None, 'test': None, 'predict': None}
    target_transforms_dict = {}
    for stage in target_transforms_config.keys():
        if target_transforms_config[stage] is None:
            target_transforms_dict[stage] = None
            continue
        for name, value in target_transforms_config[stage].items():
            if type(value) is dict:
                target_transform_arguments = []
                for a, b in value.items():
                    if a == 'num_classes' and b is None:
                        b = len(classes)
                    arg = '{}={}'.format(a, b)
                    target_transform_arguments.append(arg)
                target_transform_arguments = ','.join(
                    target_transform_arguments)
                value = target_transform_arguments
            target_transforms_dict[stage] = eval('{}({})'.format(name, value))
    return target_transforms_dict


#TODO: finish create_data_module.
def create_data_module(project_parameters):
    transforms_dict = parse_transforms(
        transforms_config=project_parameters.transforms_config)
    target_transforms_dict = parse_target_transforms(
        project_parameters.target_transforms_config)
    pass


#class
class AudioLoader:
    def __init__(self, sample_rate) -> None:
        self.sample_rate = sample_rate

    def __call__(self, path) -> Any:
        sample, sample_rate = torchaudio.load(path)
        assert self.sample_rate == sample_rate, 'please check the sample_rate argument.\nsample_rate: {}\nvalid: {}'.format(
            self.sample_rate, sample_rate)
        return sample


class DigitalFilter(nn.Module):
    def __init__(self, filter_type, sample_rate, cutoff_freq) -> None:
        super().__init__()
        assert filter_type in [
            'bandpass', 'lowpass', 'highpass', None
        ], 'please check the filter_type argument.\nfilter_type: {}\nvalid: {}'.format(
            filter_type, ['bandpass', 'lowpass', 'highpass', None])
        if type(cutoff_freq) != list:
            cutoff_freq = [cutoff_freq]
        cutoff_freq = np.array(cutoff_freq)
        # check if the cutoff frequency satisfied Nyquist theorem
        assert not any(
            cutoff_freq / (sample_rate * 0.5) > 1
        ), 'please check the cutoff_freq argument.\ncutoff_freq: {}\nvalid: {}'.format(
            cutoff_freq, [1, sample_rate // 2])
        self.filter_type = filter_type
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq

    def __call__(self, waveform):
        if self.filter_type is None or self.filter_type == 'None':
            return waveform
        elif self.filter_type == 'bandpass':
            waveform = lowpass_biquad(waveform=waveform,
                                      sample_rate=self.sample_rate,
                                      cutoff_freq=max(self.cutoff_freq))
            waveform = highpass_biquad(waveform=waveform,
                                       sample_rate=self.sample_rate,
                                       cutoff_freq=min(self.cutoff_freq))
        elif self.filter_type == 'lowpass':
            waveform = lowpass_biquad(waveform=waveform,
                                      sample_rate=self.sample_rate,
                                      cutoff_freq=max(self.cutoff_freq))
        elif self.filter_type == 'highpass':
            waveform = highpass_biquad(waveform=waveform,
                                       sample_rate=self.sample_rate,
                                       cutoff_freq=min(self.cutoff_freq))
        return waveform


class PadWaveform(nn.Module):
    def __init__(self, max_waveform_length) -> None:
        super().__init__()
        self.max_waveform_length = max_waveform_length

    def forward(self, waveform):
        # the dimension of waveform is (channels, length)
        channels, length = waveform.shape
        diff = self.max_waveform_length - length
        if diff >= 0:
            pad = (int(np.ceil(diff / 2)), int(np.floor(diff / 2)))
            waveform = F.pad(input=waveform, pad=pad)
        else:
            waveform = waveform[:, :self.max_waveform_length]
        return waveform


class OneHotEncoder:
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes

    def __call__(self, target) -> Any:
        return np.eye(self.num_classes)[target]


class LabelSmoothing(OneHotEncoder):
    def __init__(self, alpha, num_classes) -> None:
        super().__init__(num_classes=num_classes)
        self.alpha = alpha

    def __call__(self, target) -> Any:
        target = super().__call__(target)
        return (1 - self.alpha) * target + (self.alpha / self.num_classes)


class MyMNIST(MNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(len(self.data)),
                                  k=max_samples)
            self.data = self.data[index]
            self.targets = self.targets[index]


class MyCIFAR10(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(len(self.data)),
                                  k=max_samples)
            self.data = self.data[index]
            self.targets = np.array(self.targets)[index]


class MySPEECHCOMMANDS(SPEECHCOMMANDS):
    def __init__(self,
                 root: Union[str, Path],
                 loader,
                 transform,
                 target_transform,
                 download: bool = False,
                 subset: Optional[str] = None) -> None:
        super().__init__(root, download=download, subset=subset)
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [
            'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
            'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn',
            'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right',
            'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
            'visual', 'wow', 'yes', 'zero'
        ]
        self.class_to_idx = {v: idx for idx, v in enumerate(self.classes)}

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            self._walker = random.sample(population=self._walker,
                                         k=max_samples)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        path = self._walker[n]
        sample = self.loader(path)
        relpath = os.path.relpath(path, self._path)
        label, filename = os.path.split(relpath)
        target = self.class_to_idx[label]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class MyImageFolder(ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform,
                         loader=Image.open)

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            self.samples = random.sample(population=self.samples,
                                         k=max_samples)


class MyAudioFolder(DatasetFolder):
    def __init__(self,
                 root: str,
                 loader: Callable[[str], Any],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root,
                         loader=loader,
                         extensions=('.wav'),
                         transform=transform,
                         target_transform=target_transform)

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            self.samples = random.sample(population=self.samples,
                                         k=max_samples)


#TODO: finish MyLightningDataModule.
class BaseLightningDataModule(LightningDataModule):
    def __init__(self, project_parameters, transforms_dict,
                 target_transforms_dict):
        self.root = project_parameters.root
        assert project_parameters.predefined_dataset in [
            'MNIST', 'CIFAR10', 'SPEECHCOMMANDS', None
        ], 'please check the predefined_dataset argument.\npredefined_dataset: {}\nvalid: {}'.format(
            project_parameters.predefined_dataset,
            ['MNIST', 'CIFAR10', 'SPEECHCOMMANDS', None])
        self.predefined_dataset = project_parameters.predefined_dataset
        self.classes = project_parameters.classes
        self.max_samples = project_parameters.max_samples
        self.batch_size = project_parameters.batch_size
        self.num_workers = project_parameters.num_workers
        assert project_parameters.device in [
            'cpu', 'cuda'
        ], 'please check the device argument.\ndevice: {}\nvalid: {}'.format(
            project_parameters.device, ['cpu', 'cuda'])
        self.pin_memory = project_parameters.device == 'cuda' and torch.cuda.is_available(
        )
        self.transforms_dict = transforms_dict
        self.target_transforms_dict = target_transforms_dict
        self.val_size = 0.2

    def train_dataloader(
            self
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


class ImageLightningDataModule(BaseLightningDataModule):
    def __init__(self, project_parameters, transforms_dict,
                 target_transforms_dict):
        super().__init__(project_parameters, transforms_dict,
                         target_transforms_dict)

    def prepare_data(self) -> None:
        # download predefined dataset
        root = join(self.root, self.predefined_dataset)
        eval(
            'My{}(root=root, train=True, transform=None, target_transform=None, download=True)'
            .format(self.predefined_dataset))
        eval(
            'My{}(root=root, train=False, transform=None, target_transform=None, download=True)'
            .format(self.predefined_dataset))

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            if self.predefined_dataset in ['MNIST', 'CIFAR10']:
                # load predefined dataset
                train_dataset = eval(
                    'My{}(root=root, train=True, transform=self.transforms_dict["train"], target_transform=self.target_transforms_dict["train"], download=True)'
                    .format(self.predefined_dataset))
                assert self.classes == train_dataset.classes, 'please check the classes argument.\nclasses: {}\nvalid: {}'.format(
                    self.classes, train_dataset.classes)
                train_dataset.decrease_samples(max_samples=self.max_samples)
                lengths = (np.array([1 - self.val_size, self.val_size]) *
                           len(train_dataset)).astype(int)
                self.train_dataset, self.val_dataset = random_split(
                    dataset=train_dataset, lengths=lengths)
            else:
                # load dataset from root
                self.train_dataset = MyImageFolder(
                    root=join(self.root, 'train'),
                    transform=self.transforms_dict['train'],
                    target_transform=self.target_transforms_dict['train'])
                assert self.classes == self.train_dataset.classes, 'please check the classes argument.\nclasses: {}\nvalid: {}'.format(
                    self.classes, self.train_dataset.classes)
                self.train_dataset.decrease_samples(
                    max_samples=self.max_samples)
                self.val_dataset = MyImageFolder(
                    root=join(self.root, 'val'),
                    transform=self.transforms_dict['val'],
                    target_transform=self.target_transforms_dict['val'])
                self.val_dataset.decrease_samples(max_samples=self.max_samples)

        if stage == 'test' or stage is None:
            if self.predefined_dataset in ['MNIST', 'CIFAR10']:
                # load predefined dataset
                root = join(self.root, self.predefined_dataset)
                self.test_dataset = eval(
                    'My{}(root=root, train=False, transform=self.transforms_dict["test"], target_transform=self.target_transforms_dict["test"], download=True)'
                    .format(self.predefined_dataset))
                self.test_dataset.decrease_samples(
                    max_samples=self.max_samples)
            else:
                # load dataset from root
                self.test_dataset = MyImageFolder(
                    root=join(self.root, 'test'),
                    transform=self.transforms_dict['test'],
                    target_transform=self.target_transforms_dict['test'])
                self.test_dataset.decrease_samples(
                    max_samples=self.max_samples)


class AudioLightningDataModule(BaseLightningDataModule):
    def __init__(self, project_parameters, transforms_dict,
                 target_transforms_dict):
        super().__init__(project_parameters, transforms_dict,
                         target_transforms_dict)
        self.loader = AudioLoader(sample_rate=project_parameters.sample_rate)

    def prepare_data(self) -> None:
        # download predefined dataset
        root = join(self.root, self.predefined_dataset)
        makedirs(name=root, exist_ok=True)
        MySPEECHCOMMANDS(root=root,
                         loader=None,
                         transform=None,
                         target_transform=None,
                         download=True,
                         subset=None)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            if self.predefined_dataset in ['SPEECHCOMMANDS']:
                # load predefined dataset
                root = join(self.root, self.predefined_dataset)
                self.train_dataset = MySPEECHCOMMANDS(
                    root=root,
                    loader=self.loader,
                    transform=self.transforms_dict['train'],
                    target_transform=self.target_transforms_dict['train'],
                    download=True,
                    subset='training')
                assert self.classes == self.train_dataset, 'please check the classes argument.\nclasses: {}\nvalid: {}'.format(
                    self.classes, self.train_dataset)
                self.train_dataset.decrease_samples(
                    max_samples=self.max_samples)
                self.val_dataset = MySPEECHCOMMANDS(
                    root=root,
                    loader=self.loader,
                    transform=self.transforms_dict['val'],
                    target_transform=self.target_transforms_dict['val'],
                    download=True,
                    subset='validation')
                self.val_dataset.decrease_samples(max_samples=self.max_samples)
            else:
                # load dataset from root
                self.train_dataset = MyAudioFolder(
                    root=join(self.root, 'train'),
                    loader=self.loader,
                    transform=self.transforms_dict['train'],
                    target_transform=self.target_transforms_dict['train'])
                assert self.classes == self.train_dataset, 'please check the classes argument.\nclasses: {}\nvalid: {}'.format(
                    self.classes, self.train_dataset)
                self.train_dataset.decrease_samples(
                    max_samples=self.max_samples)
                self.val_dataset = MyAudioFolder(
                    root=join(self.root, 'val'),
                    loader=self.loader,
                    transform=self.transforms_dict['val'],
                    target_transform=self.target_transforms_dict['val'])
                self.val_dataset.decrease_samples(max_samples=self.max_samples)

        if stage == 'test' or stage is None:
            if self.predefined_dataset in ['SPEECHCOMMANDS']:
                # load predefined dataset
                root = join(self.root, self.predefined_dataset)
                self.test_dataset = MySPEECHCOMMANDS(
                    root=root,
                    loader=self.loader,
                    transform=self.transforms_dict['test'],
                    target_transform=self.target_transforms_dict['test'],
                    download=True,
                    subset='testing')
                self.test_dataset.decrease_samples(
                    max_samples=self.max_samples)
            else:
                # load dataset from root
                self.test_dataset = MyAudioFolder(
                    root=join(self.root, 'test'),
                    loader=self.loader,
                    transform=self.transforms_dict['test'],
                    target_transform=self.target_transforms_dict['test'])
                self.test_dataset.decrease_samples(
                    max_samples=self.max_samples)


if __name__ == '__main__':
    # project parameters
    project_parameters = {
        'transforms_config': {
            'train': {
                'Resize': [224, 224],
                'ColorJitter': None,
                'RandomRotation': 90,
                'RandomHorizontalFlip': None,
                'RandomVerticalFlip': None,
                'ToTensor': None,
                'RandomErasing': None
            },
            'val': {
                'Resize': [224, 224],
                'ColorJitter': None,
                'RandomRotation': 90,
                'RandomHorizontalFlip': None,
                'RandomVerticalFlip': None,
                'ToTensor': None
            },
            'test': {
                'Resize': [224, 224],
                'ColorJitter': None,
                'RandomRotation': 90,
                'RandomHorizontalFlip': None,
                'RandomVerticalFlip': None,
                'ToTensor': None
            },
            'predict': {
                'Resize': [224, 224],
                'ColorJitter': None,
                'RandomRotation': 90,
                'RandomHorizontalFlip': None,
                'RandomVerticalFlip': None,
                'ToTensor': None
            }
        },
        'target_transforms_config': {
            'train': {
                'LabelSmoothing': {
                    'alpha': 0.2,
                    'num_classes': None
                }
            },
            'val': {
                'OneHotEncoder': {
                    'num_classes': None
                }
            },
            'test': {
                'OneHotEncoder': {
                    'num_classes': None
                }
            }
        }
    }
    project_parameters = argparse.Namespace(**project_parameters)

    #