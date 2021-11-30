#import
import argparse
from typing import Any, Optional, Callable, Union, Tuple
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


#def
def parse_transforms(transforms_config):
    if transforms_config == 'None' or transforms_config is None:
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


#class


class DigitalFilter(nn.Module):
    def __init__(self, filter_type, sample_rate, cutoff_freq) -> None:
        super().__init__()
        assert filter_type in [
            'bandpass', 'lowpass', 'highpass', None
        ], 'please check the filter type is correct.\nfilter_type: {}\nvalid: {}'.format(
            filter_type, ['bandpass', 'lowpass', 'highpass', None])
        if type(cutoff_freq) != list:
            cutoff_freq = [cutoff_freq]
        cutoff_freq = np.array(cutoff_freq)
        # check if the cutoff frequency satisfied Nyquist theorem
        assert not any(
            cutoff_freq / (sample_rate * 0.5) > 1
        ), 'please check the cutoff frequency. the cutoff frequency value is {}.'.format(
            cutoff_freq)
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
class MyLightningDataModule(LightningDataModule):
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
        }
    }
    project_parameters = argparse.Namespace(**project_parameters)

    #