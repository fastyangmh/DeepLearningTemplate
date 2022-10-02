# import
from typing import Optional, Union, List, Dict, Callable, Any
import numpy as np
from collections import defaultdict
import torchvision
import torchaudio
from pytorch_lightning import LightningDataModule
import torch
from os.path import join
from torch.utils.data import random_split, DataLoader, Dataset
from os import makedirs
import albumentations
from abc import ABC, abstractmethod
from copy import deepcopy
try:
    from . import selfdefined_transforms
    from .dataset import PREDEFINED_DATASET, IMG_EXTENSIONS, AUDIO_EXTENSIONS, SERIES_EXTENSIONS
except:
    import selfdefined_transforms
    from dataset import PREDEFINED_DATASET, IMG_EXTENSIONS, AUDIO_EXTENSIONS, SERIES_EXTENSIONS


# def
def parse_transforms(transforms_config: Dict):
    if transforms_config is None:
        return {'train': None, 'val': None, 'test': None, 'predict': None}
    transforms_dict = defaultdict(list)
    for stage in transforms_config.keys():
        if transforms_config[stage] is None:
            transforms_dict[stage] = None
            continue
        compose_type = []
        for name, value in transforms_config[stage].items():
            #transform name
            transform_type, name = name.split('.')
            if transform_type == 'albumentations' and name in dir(
                    albumentations):
                name = f'albumentations.{name}'
            elif transform_type == 'torchvision' and name in dir(
                    torchvision.transforms):
                name = f'torchvision.transforms.{name}'
            elif transform_type == 'torchaudio' and name in dir(
                    torchaudio.transforms):
                name = f'torchaudio.transforms.{name}'
            elif transform_type == 'selfdefined' and name in dir(
                    selfdefined_transforms):
                #the transformations I have defined in this file
                name = f'selfdefined_transforms.{name}'
            else:
                assert False, f'please check the transform name in {stage} transforms_config.\n the error name: {transform_type}.{name}'
            compose_type.append(transform_type)

            #transform value
            if value is None:
                transforms_dict[stage].append(eval(f'{name}()'))
            else:
                if isinstance(value, dict):
                    transform_arguments = []
                    for a, b in value.items():
                        if isinstance(b, str):
                            arg = f'{a}="{b}"'
                        else:
                            arg = f'{a}={b}'
                        transform_arguments.append(arg)
                    transform_arguments = ','.join(transform_arguments)
                    value = transform_arguments
                transforms_dict[stage].append(eval(f'{name}({value})'))
        if 'albumentations' in compose_type:
            transforms_dict[stage] = albumentations.Compose(
                transforms_dict[stage],
                bbox_params=albumentations.BboxParams(format='yolo'))
        else:
            transforms_dict[stage] = torchvision.transforms.Compose(
                transforms_dict[stage])
    return transforms_dict


def yolo_collate_fn(batch):
    tensors, targets = zip(*batch)
    for idx, v in enumerate(targets):
        if len(v):
            v[:, 0] = idx
    tensors = np.stack(arrays=tensors, axis=0)
    targets = np.concatenate(targets, 0)
    return torch.from_numpy(tensors), torch.from_numpy(targets)


# class
class BaseLightningDataModule(LightningDataModule, ABC):
    def __init__(self, root: str, predefined_dataset: str,
                 dataset_cls: Callable, transforms_config: Dict,
                 target_transforms_config: Dict, max_samples: int,
                 classes: List, batch_size: int, num_workers: int, accelerator: str,
                 random_seed: int, **kwargs):
        super().__init__()
        assert predefined_dataset in PREDEFINED_DATASET, f'please check the predefined_dataset argument.\npredefined_dataset: {predefined_dataset}\nvalid: {PREDEFINED_DATASET}'
        self.predefined_dataset = predefined_dataset
        assert accelerator in [
            'cpu', 'gpu'
        ], f'please check the accelerator argument.\ndevice: {accelerator}\nvalid: {["cpu","gpu"]}'
        self.pin_memory = accelerator == 'gpu' and torch.cuda.is_available()
        self.root = root
        self.dataset_cls = dataset_cls
        self.transforms_dict = parse_transforms(
            transforms_config=transforms_config)
        self.target_transforms_dict = parse_transforms(
            transforms_config=target_transforms_config)
        self.max_samples = max_samples
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.val_size = 0.2

    @abstractmethod
    def prepare_data(self) -> None:
        return NotImplementedError

    @abstractmethod
    def initial_predefined_dataset(self, train: bool):
        return NotImplementedError

    def get_predefined_dataset(self, train: bool):
        if train:
            train_dataset = self.initial_predefined_dataset(train=train)
            assert self.classes == train_dataset.classes, f'please check the classes argument.\nclasses: {self.classes}\nvalid: {train_dataset.classes}'
            train_dataset.decrease_samples(max_samples=self.max_samples)
            val_dataset = deepcopy(x=train_dataset)
            val_dataset.transform = self.transforms_dict['val']
            val_dataset.target_transform = self.target_transforms_dict['val']
            self.train_dataset = self.split_dataset(dataset=train_dataset)
            self.val_dataset = self.split_dataset(dataset=val_dataset,
                                                  train=False)
        else:
            self.test_dataset = self.initial_predefined_dataset(train=train)
            self.test_dataset.decrease_samples(max_samples=self.max_samples)

    @abstractmethod
    def initial_dataset(self, stage: str):
        return NotImplementedError

    def get_dataset(self, stage: str):
        dataset = self.initial_dataset(stage=stage)
        dataset.decrease_samples(max_samples=self.max_samples)
        if stage == 'train':
            assert self.classes == dataset.classes, f'please check the classes argument.\nclasses: {self.classes}\nvalid: {dataset.classes}'
        return dataset

    def get_splits(self, len_dataset: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_size, int):
            train_len = len_dataset - self.val_size
            splits = [train_len, self.val_size]
        elif isinstance(self.val_size, float):
            val_len = int(self.val_size * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_size)}")

        return splits

    def split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = self.get_splits(len_dataset)
        dataset_train, dataset_val = random_split(
            dataset=dataset,
            lengths=splits,
            generator=torch.Generator().manual_seed(self.random_seed))

        if train:
            return dataset_train
        return dataset_val

    def set_train_and_val_dataset_for_test(self):
        try:
            #if train_dataset and val_dataset instance is torch.utils.data.dataset.Subset
            self.train_dataset.dataset.transform = self.transforms_dict['test']
            self.train_dataset.dataset.target_transform = self.target_transforms_dict[
                'test']
            self.val_dataset.dataset.transform = self.transforms_dict['test']
            self.val_dataset.dataset.target_transform = self.target_transforms_dict[
                'test']
        except:
            self.train_dataset.transform = self.transforms_dict['test']
            self.train_dataset.target_transform = self.target_transforms_dict[
                'test']
            self.val_dataset.transform = self.transforms_dict['test']
            self.val_dataset.target_transform = self.target_transforms_dict[
                'test']

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            if self.predefined_dataset is not None:
                self.get_predefined_dataset(train=True)
            else:
                self.train_dataset = self.get_dataset(stage='train')
                self.val_dataset = self.get_dataset(stage='val')

        if stage == 'test' or stage is None:
            if stage == 'test':
                self.set_train_and_val_dataset_for_test()
            if self.predefined_dataset is not None:
                self.get_predefined_dataset(train=False)
            else:
                self.test_dataset = self.get_dataset(stage='test')

    def data_loader(self,
                    dataset: Dataset,
                    shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return self.data_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self, *args: Any,
                       **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        return self.data_loader(self.val_dataset)

    def test_dataloader(self, *args: Any,
                        **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        return self.data_loader(self.test_dataset)


class ImageLightningDataModule(BaseLightningDataModule):
    def __init__(self, root: str, predefined_dataset: str,
                 dataset_cls: Callable, transforms_config: Dict,
                 target_transforms_config: Dict, max_samples: int,
                 classes: List, batch_size: int, num_workers: int, accelerator: str,
                 **kwargs):
        super().__init__(root, predefined_dataset, dataset_cls,
                         transforms_config, target_transforms_config,
                         max_samples, classes, batch_size, num_workers, accelerator,
                         **kwargs)

    def prepare_data(self) -> None:
        if self.predefined_dataset is not None:
            root = join(self.root, self.predefined_dataset)
            makedirs(name=root, exist_ok=True)
            self.dataset_cls(root=root,
                             train=True,
                             transform=None,
                             target_transform=None,
                             download=True)
            self.dataset_cls(root=root,
                             train=False,
                             transform=None,
                             target_transform=None,
                             download=True)

    def initial_predefined_dataset(self, train: bool):
        stage = 'train' if train else 'test'
        root = join(self.root, self.predefined_dataset)
        return self.dataset_cls(
            root=root,
            train=train,
            transform=self.transforms_dict[stage],
            target_transform=self.target_transforms_dict[stage],
            download=False)

    def initial_dataset(self, stage: str):
        root = join(self.root, stage)
        return self.dataset_cls(
            root=root,
            transform=self.transforms_dict[stage],
            target_transform=self.target_transforms_dict[stage],
        )


class AudioLightningDataModule(BaseLightningDataModule):
    def __init__(self, root: str, predefined_dataset: str,
                 dataset_cls: Callable, transforms_config: Dict,
                 target_transforms_config: Dict, max_samples: int,
                 classes: List, batch_size: int, num_workers: int, accelerator: str,
                 random_seed: int, **kwargs):
        super().__init__(root, predefined_dataset, dataset_cls,
                         transforms_config, target_transforms_config,
                         max_samples, classes, batch_size, num_workers, accelerator,
                         random_seed, **kwargs)
        self.loader = kwargs.get('loader')

    def prepare_data(self) -> None:
        if self.predefined_dataset is not None:
            root = join(self.root, self.predefined_dataset)
            makedirs(name=root, exist_ok=True)
            self.dataset_cls(root=root,
                             loader=None,
                             transform=None,
                             target_transform=None,
                             download=True,
                             subset=None)

    def initial_predefined_dataset(self, train: bool):
        stage = 'train' if train else 'test'
        subset = 'training' if train else 'testing'
        root = join(self.root, self.predefined_dataset)
        return self.dataset_cls(
            root=root,
            loader=self.loader,
            transform=self.transforms_dict[stage],
            target_transform=self.target_transforms_dict[stage],
            download=True,
            subset=subset)

    def initial_dataset(self, stage: str):
        root = join(self.root, stage)
        return self.dataset_cls(
            root=root,
            loader=self.loader,
            extensions=AUDIO_EXTENSIONS,
            transform=self.transforms_dict[stage],
            target_transform=self.target_transforms_dict[stage])


class SeriesLightningDataModule(BaseLightningDataModule):
    def __init__(self, root: str, predefined_dataset: str,
                 dataset_cls: Callable, transforms_config: Dict,
                 target_transforms_config: Dict, max_samples: int,
                 classes: List, batch_size: int, num_workers: int, accelerator: str,
                 random_seed: int, **kwargs):
        super().__init__(root, predefined_dataset, dataset_cls,
                         transforms_config, target_transforms_config,
                         max_samples, classes, batch_size, num_workers, accelerator,
                         random_seed, **kwargs)
        self.loader = kwargs.get('loader')

    def prepare_data(self) -> None:
        pass

    def initial_predefined_dataset(self, train: bool):
        stage = 'train' if train else 'test'
        return self.dataset_cls(
            train=train,
            transform=self.transforms_dict[stage],
            target_transform=self.target_transforms_dict[stage])

    def initial_dataset(self, stage: str):
        root = join(self.root, stage)
        return self.dataset_cls(
            root=root,
            loader=self.loader,
            extensions=SERIES_EXTENSIONS,
            transform=self.transforms_dict[stage],
            target_transform=self.target_transforms_dict[stage])


class YOLOImageLightningDataModule(ImageLightningDataModule):
    def __init__(self, root: str, predefined_dataset: str,
                 dataset_cls: Callable, transforms_config: Dict,
                 target_transforms_config: Dict, max_samples: int,
                 classes: List, batch_size: int, num_workers: int, accelerator: str,
                 **kwargs):
        super().__init__(root, predefined_dataset, dataset_cls,
                         transforms_config, target_transforms_config,
                         max_samples, classes, batch_size, num_workers, accelerator,
                         **kwargs)

    def data_loader(self,
                    dataset: Dataset,
                    shuffle: bool = False) -> DataLoader:
        data_loader = super().data_loader(dataset, shuffle)
        data_loader.collate_fn = yolo_collate_fn
        return data_loader