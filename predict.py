# import
from torch.utils.data import Dataset
from glob import glob
from os.path import join
from typing import TypeVar
T_co = TypeVar('T_co', covariant=True)


# class
class BaseDataset(Dataset):
    def __init__(self, root, extensions, loader, transform) -> None:
        super().__init__()
        samples = []
        for ext in extensions:
            samples += glob(join(root, '*{}'.format(ext)))
        assert len(
            samples
        ), 'please check if the root and extensions argument. there does not exist any files with extension in the root.\nroot: {}\nextensions: {}'.format(
            root, extensions)
        self.samples = samples
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def get_sample(self, path):
        return self.loader(path)

    def __getitem__(self, index) -> T_co:
        path = self.samples[index]
        sample = self.get_sample(path=path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class ImagePredictDataset(BaseDataset):
    def __init__(self, root, loader, transform, color_space) -> None:
        super().__init__(root=root,
                         extensions=('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                                     '.pgm', '.tif', '.tiff', '.webp'),
                         loader=loader,
                         transform=transform)
        self.color_space = color_space

    def get_sample(self, path):
        sample = super().get_sample(path)
        return sample.convert(mode=self.color_space)


class AudioPredictDataset(BaseDataset):
    def __init__(self, root, loader, transform) -> None:
        super().__init__(root=root,
                         extensions=('.wav'),
                         loader=loader,
                         transform=transform)