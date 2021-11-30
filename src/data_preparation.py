#import
import argparse
from typing import Any, Optional, Callable, Union, Tuple
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
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
                    # TODO: if b is a string, an error will occur, and the type of b needs to be checked.
                    value = ['{}={}'.format(a, b) for a, b in value.items()]
                    value = ','.join(value)
                transforms_dict[stage].append(
                    eval('{}({})'.format(name, value)))
        transforms_dict[stage] = torchvision.transforms.Compose(
            transforms_dict[stage])
    return transforms_dict


#class
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


class DigitalFilter(nn.Module):
    def __init__(self, filter_type, sample_rate, cutoff_freq) -> None:
        super().__init__()
        filter_type = {
            0: None,
            1: 'bandpass',
            2: 'lowpass',
            3: 'highpass'
        }[filter_type]
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
                'ToTensor': None
            },
            'test': {
                'Resize': [224, 224],
                'ToTensor': None
            },
            'predict': {
                'Resize': [224, 224],
                'ToTensor': None
            }
        }
    }
    project_parameters = argparse.Namespace(**project_parameters)

    #
    transforms_dict = parse_transforms(
        transforms_config=project_parameters.transforms_config)
