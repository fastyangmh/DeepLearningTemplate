#import
from typing import Callable, Dict, Tuple, Optional, TypeVar, Union, Any
import random
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, DatasetFolder, VOCSegmentation, VOCDetection, folder
import random
from torchaudio.datasets import SPEECHCOMMANDS, CMUARCTIC
from pathlib import Path
from torch import Tensor
import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.datasets import load_breast_cancer
from glob import glob
import numpy as np
import pandas as pd
import inspect
import sys
import torchaudio

T_co = TypeVar('T_co', covariant=True)


#class
class AudioLoader:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    def __call__(self, path: str) -> Any:
        sample, sample_rate = torchaudio.load(path)
        if self.sample_rate != sample_rate:
            print(
                f'please check the sample_rate argument, although the waveform will automatically be resampled, you should check the sample_rate argument.\nsample_rate: {self.sample_rate}\nvalid: {sample_rate}'
            )
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

    def decrease_samples(self, max_samples: int):
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

    def decrease_samples(self, max_samples: int):
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
        super().__init__()
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

    def __getitem__(self, index: int) -> T_co:
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img = np.array(img)
            target = np.array(target)
            transformed = self.transform(image=img, mask=target)
            img = transformed['image']
            target = transformed['mask']
        target[target == 255] = 0
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def decrease_samples(self, max_samples: int):
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

    def get_bboxes(self, target: Dict):
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

    def __getitem__(self, index: int) -> T_co:
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

    def decrease_samples(self, max_samples: int):
        if max_samples is not None:
            index = random.sample(population=range(len(self.dataset)),
                                  k=max_samples)
            self.dataset.images = np.array(self.dataset.images)[index]
            self.dataset.targets = np.array(self.dataset.targets)[index]


class MySPEECHCOMMANDS(SPEECHCOMMANDS):
    def __init__(self,
                 root: Union[str, Path],
                 loader: Callable[[str], Any],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
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

    def decrease_samples(self, max_samples: int):
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
                 root: Union[str, Path],
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 subset: Optional[str] = None,
                 loader: Callable[[str], Any] = None,
                 target_transform: Optional[Callable] = None) -> None:
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

    def __getitem__(self, index: int):
        sample1 = self.male_dataset[index][0]
        sample2 = self.female_dataset[index][0]
        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        return sample1, sample2

    def decrease_samples(self, max_samples: int):
        if max_samples is not None:
            index = random.sample(population=range(len(self.male_dataset)),
                                  k=max_samples)
            self.male_dataset._walker = np.array(
                self.male_dataset._walker)[index]
            self.female_dataset._walker = np.array(
                self.female_dataset._walker)[index]


class MyBreastCancerDataset(Dataset):
    # NOTE: the MyBreastCancerDataset dataset only contains training and validation datasets and the ratio is 8:2
    def __init__(self,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
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

    def __getitem__(self, index: int) -> T_co:
        sample = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def decrease_samples(self, max_samples: int):
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

    def find_files(self, filename: str):
        for ext in self.extensions:
            temp = []
            for f in sorted(glob(os.path.join(self.root,
                                              f'{filename}*{ext}'))):
                temp.append(self.loader(f))
        return pd.concat(temp).values

    def find_classes(self, filename: str):
        return list(
            np.loadtxt(os.path.join(self.root, f'{filename}'),
                       dtype=str,
                       delimiter='\n'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> T_co:
        sample = self.samples[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def decrease_samples(self, max_samples: int):
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

    def decrease_samples(self, max_samples: int):
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

    def decrease_samples(self, max_samples: int):
        if max_samples is not None:
            self.samples = random.sample(population=self.samples,
                                         k=max_samples)


#global parameters
GENRAL_DATASET = ['MyAudioFolder', 'MyImageFolder', 'MySeriesFolder']
PREDEFINED_DATASET = [
    name[2:]
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if 'My' in name and name not in GENRAL_DATASET
] + [None]
IMG_EXTENSIONS = folder.IMG_EXTENSIONS
AUDIO_EXTENSIONS = ('.wav', '.flac')
SERIES_EXTENSIONS = ('.csv')
