#import
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Tuple, TypeVar, Any, Union
from glob import glob
from os.path import join, splitext, isfile
import argparse
import torch
from tqdm import tqdm

T_co = TypeVar('T_co', covariant=True)
try:
    from . import create_model, parse_transforms, IMG_EXTENSIONS, AUDIO_EXTENSIONS, SERIES_EXTENSIONS
except:
    from data_preparation import parse_transforms
    from dataset import IMG_EXTENSIONS, AUDIO_EXTENSIONS, SERIES_EXTENSIONS
    from model import create_model


#class
class PredictDataset(Dataset):
    def __init__(self, root: str, extensions: Tuple, loader: Callable,
                 transform: Callable) -> None:
        super().__init__()
        if isfile(root):
            samples = [root]
        else:
            samples = [
                f for f in glob(join(root, '*'))
                if splitext(f)[-1] in extensions
            ]
        assert len(
            samples
        ), f'please check the root and extensions argument. there does not exist any files with extension in the root.\nroot: {root}\nextensions: {extensions}'
        self.samples = sorted(samples)
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> T_co:
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class Predictor():
    def __init__(self, project_parameters: argparse.Namespace,
                 loader: Callable) -> None:
        self.model = create_model(project_parameters=project_parameters).eval()
        if project_parameters.accelerator == 'cuda' and torch.cuda.is_available(
        ):
            self.model = self.model.to(project_parameters.accelerator)
            pin_memory = True
        else:
            pin_memory = False
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.loader = loader
        self.accelerator = project_parameters.accelerator
        self.batch_size = project_parameters.batch_size
        self.num_workers = project_parameters.num_workers
        if project_parameters.file_extensions in [
                'IMG_EXTENSIONS', 'AUDIO_EXTENSIONS', 'SERIES_EXTENSIONS'
        ]:
            self.extensions = eval(project_parameters.file_extensions)
        else:
            self.extensions = project_parameters.file_extensions
        self.pin_memory = pin_memory

    def __call__(self, inputs: Union[str, torch.Tensor]) -> Any:
        result = []
        if isinstance(inputs, str):
            dataset = PredictDataset(root=inputs,
                                     extensions=self.extensions,
                                     loader=self.loader,
                                     transform=self.transform)
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory)
            with torch.no_grad():
                for x in tqdm(data_loader):
                    if self.pin_memory:
                        x = x.to(self.accelerator)
                    result.append(self.model(x).tolist())
        elif isinstance(inputs, torch.Tensor):
            x = self.transform(inputs)
            if self.pin_memory:
                x = x.to(self.accelerator)
            with torch.no_grad():
                result.append(self.model(x[None]).tolist())
        else:
            assert 0, f'please check the inputs data type.\n the inputs data type: {type(inputs)}\nvalid: {["str","torch.Tensor"]}'
        return np.concatenate(result, 0)
