# used but changed from: https://github.com/lucidrains/gigagan-pytorch

from functools import partial
from pathlib import Path
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms as T

from beartype.door import is_bearable
from beartype.typing import Tuple

# helper functions

def exists(val):
    return val is not None

def convert_image_to_fn(img_type, image):
    if image.mode == img_type:
        return image

    return image.convert(img_type)

# custom collation function
# so dataset can return a str and it will collate into List[str]

def collate_tensors_or_str(data):
    is_one_data = not isinstance(data[0], tuple)

    if is_one_data:
        data = torch.stack(data)
        return (data,)

    outputs = []
    for datum in zip(*data):
        if is_bearable(datum, Tuple[str, ...]):
            output = list(datum)
        else:
            output = torch.stack(datum)

        outputs.append(output)

    return tuple(outputs)

# dataset classes

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        assert len(self.paths) > 0, 'your folder contains no images'
        assert len(self.paths) > 100, 'you need at least 100 images, 10k for research paper, millions for miraculous results (try Laion-5B)'

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, shuffle = True, drop_last = True, **kwargs)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class TextImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        channels = 3
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels

        self.image_paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.text_paths = [p for ext in ['csv'] for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        assert len(self.image_paths) > 0, 'your folder contains no images'
        assert len(self.text_paths) > 0, 'your folder contains no csvs'

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(path)

        if len(img.getbands()) != 3:
            index = index+1
            path = self.image_paths[index]
            img = Image.open(path)
        
        # grab the corresponding csv containing the text
        csv = [x for x in self.text_paths if path.parent == x.parent][0]
        image_num = path.stem.split('_')[1]
        df = pd.read_csv(csv)
        text = df.loc[int(image_num), 'TEXT']
        return self.transform(img), text

class MockTextImageDataset(TextImageDataset):
    def __init__(
        self,
        image_size,
        length = int(1e5),
        channels = 3
    ):
        self.image_size = image_size
        self.channels = channels
        self.length = length

    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        mock_image = torch.randn(self.channels, self.image_size, self.image_size)
        return mock_image, 'mock text'
