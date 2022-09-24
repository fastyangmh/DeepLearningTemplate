# import
from typing import Any, Optional, Callable, Union, Tuple, List, Dict, TypeVar
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, DatasetFolder, VOCSegmentation, VOCDetection
import random
from torchaudio.datasets import SPEECHCOMMANDS, CMUARCTIC
from pathlib import Path
from torch import Tensor
import os
from collections import defaultdict
import torchvision
import torchaudio
from PIL import Image
from pytorch_lightning import LightningDataModule
import torch
from os.path import join
from torch.utils.data import random_split, DataLoader, Dataset
from os import makedirs
from sklearn.datasets import load_breast_cancer
from glob import glob
import pandas as pd
import albumentations
from deeplearningtemplate import selfdefined_transforms

T_co = TypeVar('T_co', covariant=True)

#parameters
PREDEFINED_DATASET = [
    'MNIST', 'CIFAR10', 'VOCSegmentation', 'SPEECHCOMMANDS',
    'BreastCancerDataset', 'CMUARCTICForVC', 'VOCDetection', None
]


# def
def parse_transforms(transforms_config):
    if transforms_config is None:
        return {'train': None, 'val': None, 'test': None, 'predict': None}
    transforms_dict = defaultdict(list)
    for stage in transforms_config.keys():
        compose_type = ''
        if transforms_config[stage] is None:
            transforms_dict[stage] = None
            continue
        compose_type = []
        for name, value in transforms_config[stage].items():
            #transform name
            transfrom_type, name = name.split('.')
            if transfrom_type == 'albumentations' and name in dir(
                    albumentations):
                name = 'albumentations.{}'.format(name)
            elif transfrom_type == 'torchvision' and name in dir(
                    torchvision.transforms):
                name = 'torchvision.transforms.{}'.format(name)
            elif transfrom_type == 'torchaudio' and name in dir(
                    torchaudio.transforms):
                name = 'torchaudio.transforms.{}'.format(name)
            elif transfrom_type == 'selfdefined' and name in dir(
                    selfdefined_transforms):
                #my defined transform in this file
                name = 'selfdefined_transforms.{}'.format(name)
            compose_type.append(transfrom_type)
            #transform value
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
        if 'albumentations' in compose_type:
            transforms_dict[stage] = albumentations.Compose(
                transforms_dict[stage],
                bbox_params=albumentations.BboxParams(format='yolo'))
        else:
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
            #transform name
            transfrom_type, name = name.split('.')
            if transfrom_type == 'selfdefined' and name in dir(
                    selfdefined_transforms):
                #my defined transform in this file
                name = 'selfdefined_transforms.{}'.format(name)
            #transform value
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


def yolo_collate_fn(batch):
    tensors, targets = zip(*batch)
    for idx, v in enumerate(targets):
        if len(v):
            v[:, 0] = idx
    tensors = np.stack(arrays=tensors, axis=0)
    targets = np.concatenate(targets, 0)
    return torch.from_numpy(tensors), torch.from_numpy(targets)


# class
class AudioLoader:
    def __init__(self, sample_rate) -> None:
        self.sample_rate = sample_rate

    def __call__(self, path) -> Any:
        sample, sample_rate = torchaudio.load(path)
        if self.sample_rate != sample_rate:
            print(
                'please check the sample_rate argument, although the waveform will automatically be resampled, you should check the sample_rate argument.\nsample_rate: {}\nvalid: {}'
                .format(self.sample_rate, sample_rate))
            sample = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate)(sample)
        return sample


class MyMNIST(MNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root=root,
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
        super().__init__(root=root,
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


class MyVOCSegmentation(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        year = '2007'
        image_set = 'train' if train else 'test'
        dataset = VOCSegmentation(root=root,
                                  year=year,
                                  image_set=image_set,
                                  download=download,
                                  transform=transform,
                                  target_transform=target_transform)
        self.images, self.masks = dataset.images, dataset.masks
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor'
        ]
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> T_co:
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        if self.transform is not None:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(new_state=state)
            target = self.transform(target)
        if target.dtype == torch.float32:
            target = (target * 255).round().long()
            target[target == 255] = 0
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(len(self.images)),
                                  k=max_samples)
            self.images = np.array(self.images)[index]
            self.masks = np.array(self.masks)[index]


class MyVOCDetection(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 image_size=None) -> None:
        super().__init__()
        year = '2007'
        image_set = 'train' if train else 'test'
        self.dataset = VOCDetection(root=root,
                                    year=year,
                                    image_set=image_set,
                                    download=download,
                                    transform=None,
                                    target_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def get_bboxes(self, target):
        size = target['annotation']['size']
        width, height, chans = int(size['width']), int(size['height']), int(
            size['depth'])
        objects = target['annotation']['object']
        bboxes = []
        for v in objects:
            name = v['name']
            xyxy = np.array([int(bndbox) for bndbox in v['bndbox'].values()])
            xywh = np.zeros_like(xyxy, dtype=float)
            xywh[0] = (xyxy[0] + xyxy[2]) / (2 * width)
            xywh[1] = (xyxy[1] + xyxy[3]) / (2 * height)
            xywh[2] = (xyxy[2] - xyxy[0]) / width
            xywh[3] = (xyxy[3] - xyxy[1]) / height
            bboxes.append([self.class_to_idx[name], *xywh])
        return np.array(bboxes)

    def __getitem__(self, index) -> T_co:
        img, target = self.dataset[
            index]  #RGB order and dimension is (width, height, channels)
        img = np.array(img)  #the img dimension is (height, width, channels)
        bboxes = self.get_bboxes(
            target=target
        )  #the bboxes dimension is (n_objects, bounding_box_info) [[c, x, y, w, h]]
        if self.transform:
            if len(bboxes) == 0:
                img, _ = self.transform(image=img,
                                        bboxes=[[1e-10] * 4 + [0]]).values()
            else:
                bboxes = bboxes[:,
                                [1, 2, 3, 4,
                                 0]]  #[[c, x, y, w, h]] -> [[x, y, w, h, c]]
                img, bboxes = self.transform(image=img, bboxes=bboxes).values()
                bboxes = np.array(bboxes)
                bboxes = bboxes[..., None] if len(bboxes) == 0 else bboxes
                #[[x, y, w, h, c]] -> [[c, x, y, w, h]]
                bboxes = bboxes[:, [4, 0, 1, 2, 3]]
            size = max(img.shape[:-1])
            assert size == self.image_size, f'the transformed image size not equal image_size in config.yml.\ntransformed image size: {size}\nimage_size in config.yml: {self.image_size}'
        if self.target_transform and len(bboxes):
            labels = bboxes[:, 0].astype(int)
            labels = self.target_transform(labels)
            bboxes = np.append(arr=labels, values=bboxes[:, 1:], axis=-1)
        if len(bboxes):
            #add target image index for build_targets()
            bboxes = np.append(arr=np.zeros(shape=(len(bboxes), 1)),
                               values=bboxes,
                               axis=-1)
        else:
            #image_index + one hot encoder length + xywh if self.target_transform
            #else image_index + c + xywh
            l = 1 + len(
                self.classes) + 4 if self.target_transform else 1 + 1 + 4
            bboxes = np.zeros(shape=(0, l))
        img = img.transpose(
            2, 0, 1)  #transpose dimension to (channels, width, height)
        bboxes = bboxes.astype(np.float32)
        return img, bboxes

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(len(self.dataset)),
                                  k=max_samples)
            self.dataset.images = np.array(self.dataset.images)[index]
            self.dataset.targets = np.array(self.dataset.targets)[index]


class MySPEECHCOMMANDS(SPEECHCOMMANDS):
    def __init__(self,
                 root: Union[str, Path],
                 loader,
                 transform,
                 target_transform,
                 download: bool = False,
                 subset: Optional[str] = None) -> None:
        super().__init__(root=root, download=download, subset=subset)
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


class MyCMUARCTICForVC(Dataset):
    def __init__(self,
                 root,
                 transform,
                 download,
                 subset,
                 loader=None,
                 target_transform=None) -> None:
        super().__init__()
        self.male_dataset = CMUARCTIC(root=root, url='aew', download=download)
        self.female_dataset = CMUARCTIC(root=root,
                                        url='slt',
                                        download=download)
        assert len(self.male_dataset) == len(
            self.female_dataset
        ), f'the male and female dataset have difference lengths.\nthe length of male dataset: {len(self.male_dataset)}\nthe length of female dataset: {len(self.female_dataset)}'
        l = len(self.male_dataset)
        if subset == 'training':
            self.male_dataset._walker = self.male_dataset._walker[:int(l *
                                                                       0.8)]
            self.female_dataset._walker = self.female_dataset._walker[:int(l *
                                                                           0.8
                                                                           )]
        else:
            self.male_dataset._walker = self.male_dataset._walker[int(l *
                                                                      0.8):]
            self.female_dataset._walker = self.female_dataset._walker[int(l *
                                                                          0.8
                                                                          ):]
        self.transform = transform
        self.classes = ['aew', 'slt']
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}

    def __len__(self):
        return len(self.male_dataset)

    def __getitem__(self, index):
        sample1 = self.male_dataset[index][0]
        sample2 = self.female_dataset[index][0]
        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        return sample1, sample2

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(len(self.male_dataset)),
                                  k=max_samples)
            self.male_dataset._walker = np.array(
                self.male_dataset._walker)[index]
            self.female_dataset._walker = np.array(
                self.female_dataset._walker)[index]


class MyBreastCancerDataset(Dataset):
    # NOTE: this dataset contains only training and validation datasets and the training and validation of ratio is 8:2
    def __init__(self, train, transform, target_transform) -> None:
        super().__init__()
        self.data = load_breast_cancer().data
        self.targets = load_breast_cancer().target
        #convert the data type of self.data and self.targets
        self.data = self.data.astype(np.float32)
        self.targets = self.targets.astype(np.int32)
        self.classes = list(load_breast_cancer().target_names)
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}
        l = len(self.data)
        if train:
            self.data = self.data[:int(l * 0.8)]
            self.targets = self.targets[:int(l * 0.8)]
        else:
            self.data = self.data[int(l * 0.8):]
            self.targets = self.targets[int(l * 0.8):]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        sample = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(len(self.data)),
                                  k=max_samples)
            self.data = self.data[index]
            self.targets = self.targets[index]


class MySeriesFolder(Dataset):
    def __init__(self,
                 root: str,
                 loader: Callable[[str], Any],
                 extensions: Optional[Tuple[str, ...]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.root = root
        self.loader = loader
        #TODO: support multiple extensions
        assert extensions.count(
            '.'
        ) == 1 and '.csv' in extensions, 'currently, only support CSV extension.'
        self.extensions = [extensions]
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self.find_files(filename='sample')
        self.targets = self.find_files(filename='target')
        #convert the data type of self.samples and self.targets
        self.samples = self.samples.astype(np.float32)
        self.targets = self.targets.astype(np.int32)
        #reduce unnecessary dimension in self.targets
        self.targets = self.targets[:, 0] if self.targets.shape[
            -1] == 1 else self.targets
        self.classes = self.find_classes(filename='classes.txt')
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}

    def find_files(self, filename):
        for ext in self.extensions:
            temp = []
            for f in sorted(glob(join(self.root, f'{filename}*{ext}'))):
                temp.append(self.loader(f))
        return pd.concat(temp).values

    def find_classes(self, filename):
        return list(
            np.loadtxt(join(self.root, f'{filename}'),
                       dtype=str,
                       delimiter='\n'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> T_co:
        sample = self.samples[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(len(self.samples)),
                                  k=max_samples)
            self.samples = self.samples[index]
            self.targets = self.targets[index]


class MyImageFolder(ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root=root,
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
                 extensions=('.wav', '.flac'),
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root=root,
                         loader=loader,
                         extensions=extensions,
                         transform=transform,
                         target_transform=target_transform)

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            self.samples = random.sample(population=self.samples,
                                         k=max_samples)


class BaseLightningDataModule(LightningDataModule):
    def __init__(self, root, predefined_dataset, classes, max_samples,
                 batch_size, num_workers, device, transforms_config,
                 target_transforms_config):
        super().__init__()
        self.root = root
        assert predefined_dataset in PREDEFINED_DATASET, 'please check the predefined_dataset argument.\npredefined_dataset: {}\nvalid: {}'.format(
            predefined_dataset, PREDEFINED_DATASET)
        self.predefined_dataset = predefined_dataset
        self.classes = classes
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        assert device in [
            'cpu', 'cuda'
        ], 'please check the device argument.\ndevice: {}\nvalid: {}'.format(
            device, ['cpu', 'cuda'])
        self.pin_memory = device == 'cuda' and torch.cuda.is_available()
        self.transforms_dict = parse_transforms(
            transforms_config=transforms_config)
        self.target_transforms_dict = parse_target_transforms(
            target_transforms_config=target_transforms_config, classes=classes)
        self.val_size = 0.2

    def set_train_and_val_dataset_transform_to_test(self):
        #maybe the dataset was split by random_split, so use try and except expression
        try:
            #set the transform and target_transform of the training dataset to test
            self.train_dataset.dataset.transform = self.transforms_dict['test']
            self.train_dataset.dataset.target_transform = self.target_transforms_dict[
                'test']
            #set the transform and target_transform of the validation dataset to test
            self.val_dataset.dataset.transform = self.transforms_dict['test']
            self.val_dataset.dataset.target_transform = self.target_transforms_dict[
                'test']
        except:
            #set the transform and target_transform of the training dataset to test
            self.train_dataset.transform = self.transforms_dict['test']
            self.train_dataset.target_transform = self.target_transforms_dict[
                'test']
            #set the transform and target_transform of the validation dataset to test
            self.val_dataset.transform = self.transforms_dict['test']
            self.val_dataset.target_transform = self.target_transforms_dict[
                'test']

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
    def __init__(self, root, predefined_dataset, classes, max_samples,
                 batch_size, num_workers, device, transforms_config,
                 target_transforms_config, dataset_class):
        super().__init__(root=root,
                         predefined_dataset=predefined_dataset,
                         classes=classes,
                         max_samples=max_samples,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         device=device,
                         transforms_config=transforms_config,
                         target_transforms_config=target_transforms_config)
        self.dataset_class = dataset_class

    def prepare_data(self) -> None:
        # download predefined dataset
        if self.predefined_dataset is not None:
            root = join(self.root, self.predefined_dataset)
            self.dataset_class(root=root,
                               train=True,
                               transform=None,
                               target_transform=None,
                               download=True)
            self.dataset_class(root=root,
                               train=False,
                               transform=None,
                               target_transform=None,
                               download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            if self.predefined_dataset is not None:
                # load predefined dataset
                root = join(self.root, self.predefined_dataset)
                train_dataset = self.dataset_class(
                    root=root,
                    train=True,
                    transform=self.transforms_dict["train"],
                    target_transform=self.target_transforms_dict["train"],
                    download=True)
                assert self.classes == train_dataset.classes, 'please check the classes argument.\nclasses: {}\nvalid: {}'.format(
                    self.classes, train_dataset.classes)
                train_dataset.decrease_samples(max_samples=self.max_samples)
                lengths = np.array([
                    np.ceil((1 - self.val_size) * len(train_dataset)),
                    np.floor(self.val_size * len(train_dataset))
                ]).astype(int)
                # it cannot assign target_transform of validation to val_dataset after random splitting,
                # because it will overwrite the target_transform of train
                self.train_dataset, self.val_dataset = random_split(
                    dataset=train_dataset, lengths=lengths)
            else:
                # load dataset from root
                self.train_dataset = self.dataset_class(
                    root=join(self.root, 'train'),
                    transform=self.transforms_dict['train'],
                    target_transform=self.target_transforms_dict['train'])
                assert self.classes == self.train_dataset.classes, 'please check the classes argument.\nclasses: {}\nvalid: {}'.format(
                    self.classes, self.train_dataset.classes)
                self.train_dataset.decrease_samples(
                    max_samples=self.max_samples)
                self.val_dataset = self.dataset_class(
                    root=join(self.root, 'val'),
                    transform=self.transforms_dict['val'],
                    target_transform=self.target_transforms_dict['val'])
                self.val_dataset.decrease_samples(max_samples=self.max_samples)

        if stage == 'test' or stage is None:
            if stage == 'test':
                self.set_train_and_val_dataset_transform_to_test()
            if self.predefined_dataset is not None:
                # load predefined dataset
                root = join(self.root, self.predefined_dataset)
                self.test_dataset = self.dataset_class(
                    root=root,
                    train=False,
                    transform=self.transforms_dict["test"],
                    target_transform=self.target_transforms_dict["test"],
                    download=True)
                self.test_dataset.decrease_samples(
                    max_samples=self.max_samples)
            else:
                # load dataset from root
                self.test_dataset = self.dataset_class(
                    root=join(self.root, 'test'),
                    transform=self.transforms_dict['test'],
                    target_transform=self.target_transforms_dict['test'])
                self.test_dataset.decrease_samples(
                    max_samples=self.max_samples)


class AudioLightningDataModule(BaseLightningDataModule):
    def __init__(self, root, predefined_dataset, classes, max_samples,
                 batch_size, num_workers, device, transforms_config,
                 target_transforms_config, sample_rate, dataset_class):
        super().__init__(root=root,
                         predefined_dataset=predefined_dataset,
                         classes=classes,
                         max_samples=max_samples,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         device=device,
                         transforms_config=transforms_config,
                         target_transforms_config=target_transforms_config)
        self.loader = AudioLoader(sample_rate=sample_rate)
        self.dataset_class = dataset_class

    def prepare_data(self) -> None:
        # download predefined dataset
        if self.predefined_dataset is not None:
            root = join(self.root, self.predefined_dataset)
            makedirs(name=root, exist_ok=True)
            self.dataset_class(root=root,
                               loader=None,
                               transform=None,
                               target_transform=None,
                               download=True,
                               subset=None)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            if self.predefined_dataset is not None:
                # load predefined dataset
                root = join(self.root, self.predefined_dataset)
                self.train_dataset = self.dataset_class(
                    root=root,
                    loader=self.loader,
                    transform=self.transforms_dict['train'],
                    target_transform=self.target_transforms_dict['train'],
                    download=True,
                    subset='training')
                assert self.classes == self.train_dataset.classes, 'please check the classes argument.\nclasses: {}\nvalid: {}'.format(
                    self.classes, self.train_dataset.classes)
                self.train_dataset.decrease_samples(
                    max_samples=self.max_samples)
                self.val_dataset = self.dataset_class(
                    root=root,
                    loader=self.loader,
                    transform=self.transforms_dict['val'],
                    target_transform=self.target_transforms_dict['val'],
                    download=True,
                    subset='validation')
                self.val_dataset.decrease_samples(max_samples=self.max_samples)
            else:
                # load dataset from root
                self.train_dataset = self.dataset_class(
                    root=join(self.root, 'train'),
                    loader=self.loader,
                    transform=self.transforms_dict['train'],
                    target_transform=self.target_transforms_dict['train'])
                assert self.classes == self.train_dataset.classes, 'please check the classes argument.\nclasses: {}\nvalid: {}'.format(
                    self.classes, self.train_dataset.classes)
                self.train_dataset.decrease_samples(
                    max_samples=self.max_samples)
                self.val_dataset = self.dataset_class(
                    root=join(self.root, 'val'),
                    loader=self.loader,
                    transform=self.transforms_dict['val'],
                    target_transform=self.target_transforms_dict['val'])
                self.val_dataset.decrease_samples(max_samples=self.max_samples)

        if stage == 'test' or stage is None:
            if stage == 'test':
                self.set_train_and_val_dataset_transform_to_test()
            if self.predefined_dataset is not None:
                # load predefined dataset
                root = join(self.root, self.predefined_dataset)
                self.test_dataset = self.dataset_class(
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
                self.test_dataset = self.dataset_class(
                    root=join(self.root, 'test'),
                    loader=self.loader,
                    transform=self.transforms_dict['test'],
                    target_transform=self.target_transforms_dict['test'])
                self.test_dataset.decrease_samples(
                    max_samples=self.max_samples)


class SeriesLightningDataModule(BaseLightningDataModule):
    def __init__(self, root, predefined_dataset, classes, max_samples,
                 batch_size, num_workers, device, transforms_config,
                 target_transforms_config, dataset_class):
        super().__init__(root, predefined_dataset, classes, max_samples,
                         batch_size, num_workers, device, transforms_config,
                         target_transforms_config)
        self.loader = pd.read_csv
        self.dataset_class = dataset_class

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            if self.predefined_dataset is not None:
                # load predefined dataset
                self.train_dataset = self.dataset_class(
                    train=True,
                    transform=self.transforms_dict['train'],
                    target_transform=self.target_transforms_dict['train'])
                assert self.classes == self.train_dataset.classes, 'please check the classes argument.\nclasses: {}\nvalid: {}'.format(
                    self.classes, self.train_dataset.classes)
                self.train_dataset.decrease_samples(
                    max_samples=self.max_samples)
                self.val_dataset = self.dataset_class(
                    train=False,
                    transform=self.transforms_dict['val'],
                    target_transform=self.target_transforms_dict['val'])
                self.val_dataset.decrease_samples(max_samples=self.max_samples)
            else:
                # load dataset from root
                self.train_dataset = self.dataset_class(
                    root=join(self.root, 'train'),
                    loader=self.loader,
                    extensions=('.csv'),
                    transform=self.transforms_dict['train'],
                    target_transform=self.target_transforms_dict['train'])
                assert self.classes == self.train_dataset.classes, 'please check the classes argument.\nclasses: {}\nvalid: {}'.format(
                    self.classes, self.train_dataset.classes)
                self.train_dataset.decrease_samples(
                    max_samples=self.max_samples)
                self.val_dataset = self.dataset_class(
                    root=join(self.root, 'val'),
                    loader=self.loader,
                    extensions=('.csv'),
                    transform=self.transforms_dict['val'],
                    target_transform=self.target_transforms_dict['val'])
                self.val_dataset.decrease_samples(max_samples=self.max_samples)

        if stage == 'test' or stage is None:
            if stage == 'test':
                self.set_train_and_val_dataset_transform_to_test()
            if self.predefined_dataset is not None:
                # load predefined dataset
                self.test_dataset = self.dataset_class(
                    train=False,
                    transform=self.transforms_dict['test'],
                    target_transform=self.target_transforms_dict['test'])
                self.test_dataset.decrease_samples(
                    max_samples=self.max_samples)
            else:
                # load dataset from root
                self.test_dataset = self.dataset_class(
                    root=join(self.root, 'test'),
                    loader=self.loader,
                    extensions=('.csv'),
                    transform=self.transforms_dict['test'],
                    target_transform=self.target_transforms_dict['test'])
                self.test_dataset.decrease_samples(
                    max_samples=self.max_samples)


class YOLOImageLightningDataModule(ImageLightningDataModule):
    def __init__(self, root, predefined_dataset, classes, max_samples,
                 batch_size, num_workers, device, transforms_config,
                 target_transforms_config, dataset_class):
        super().__init__(root, predefined_dataset, classes, max_samples,
                         batch_size, num_workers, device, transforms_config,
                         target_transforms_config, dataset_class)

    def train_dataloader(
            self
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        dataloader = super().train_dataloader()
        dataloader.collate_fn = yolo_collate_fn
        return dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        dataloader = super().val_dataloader()
        dataloader.collate_fn = yolo_collate_fn
        return dataloader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        dataloader = super().test_dataloader()
        dataloader.collate_fn = yolo_collate_fn
        return dataloader
