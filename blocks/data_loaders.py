import os
import json
import monai
import nibabel as nib

from tqdm import tqdm

import torch
from torch.utils.data import IterableDataset, WeightedRandomSampler

from core.convex_adam_utils import MINDSSC


def read_json_data_file(
        data_file_path,
        data_dir,
        keys
):
    with open(data_file_path) as f:
        json_data = json.load(f)

    files = []
    for key in keys:

        key_files = []
        json_data_training = json_data[key]
        for d in json_data_training:
            for k, v in d.items():
                if isinstance(d[k], list):
                    d[k] = [os.path.join(data_dir, iv) for iv in d[k]]
                elif isinstance(d[k], str):
                    d[k] = os.path.join(data_dir, d[k]) if len(d[k]) > 0 else d[k]
            key_files.append(d)
        files.append(key_files)

    return files


class CacheDataloader(IterableDataset):

    def __init__(self, data_list, batch_size, weights=None, mind=False):
        self.data_list = data_list
        self.batch_size = batch_size
        self.mind = mind

        self.samples = []
        self.images_cache = {}
        self.mind_cache = {}
        for data_pair in tqdm(self.data_list):

            self.samples.append(
                {
                    'image_1': data_pair['image_1'],
                    'image_2': data_pair['image_2'],
                    'idx': data_pair['idx']
                }
            )

            if data_pair['image_1'] not in self.images_cache:
                self.images_cache[data_pair['image_1']] = torch.tensor(nib.load(data_pair['image_1']).get_fdata(), dtype=torch.float).unsqueeze(0).unsqueeze(0)

                if self.mind:
                    self.mind_cache[data_pair['image_1']] = torch.nn.functional.avg_pool3d(MINDSSC(self.images_cache[data_pair['image_1']].cuda(), 1, 2), 2).cpu()

            if data_pair['image_2'] not in self.images_cache:
                self.images_cache[data_pair['image_2']] = torch.tensor(nib.load(data_pair['image_2']).get_fdata(), dtype=torch.float).unsqueeze(0).unsqueeze(0)

                if self.mind:
                    self.mind_cache[data_pair['image_1']] = torch.nn.functional.avg_pool3d(MINDSSC(self.images_cache[data_pair['image_1']].cuda(), 1, 2), 2).cpu()

        if weights is not None:
            self.weights = weights
        else:
            self.weights = None

    def generate(self):

        while True:
            idxs = torch.tensor(list(WeightedRandomSampler(self.weights, self.batch_size, replacement=True))).long()

            out = {
                'image_1': torch.concatenate([self.images_cache[self.samples[idx]['image_1']] for idx in idxs], dim=0),
                'image_2':  torch.concatenate([self.images_cache[self.samples[idx]['image_2']] for idx in idxs], dim=0),
                'idx': torch.tensor([self.samples[idx]['idx'] for idx in idxs])
            }

            if self.mind:
                out['mind_1'] = torch.concatenate([self.mind_cache[self.samples[idx]['image_1']] for idx in idxs], dim=0)
                out['mind_2'] = torch.concatenate([self.mind_cache[self.samples[idx]['image_2']] for idx in idxs], dim=0)

            yield out

    def __iter__(self):
        return iter(self.generate())


def get_data_loaders(
    settings
):

    # Get training data list
    training_data_list = read_json_data_file(
        data_file_path=settings['dataset']['data_file'],
        data_dir=settings['dataset']['data_dir'],
        keys='training'
    )[0]

    max_samples_num = settings['dataset']['max_samples_num']
    if max_samples_num > 0:
        training_data_list = training_data_list[:max_samples_num]

    # Get validation data list
    validation_data_list = read_json_data_file(
        data_file_path=settings['dataset']['data_file'],
        data_dir=settings['dataset']['data_dir'],
        keys='validation'
    )[0]

    # Instantiate training data loader that cashes data into RAM memory to speed up training
    training_data_loader = CacheDataloader(
        data_list=training_data_list,
        batch_size=settings['dataloader']['batch_size']
    )

    # Instantiate MONAI-based validation data loader
    transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=settings['dataset']['keys']),
            monai.transforms.EnsureChannelFirstd(keys=settings['dataset']['keys'], channel_dim='no_channel'),
            monai.transforms.ToTensord(keys=settings['dataset']['keys'])
        ]
    )

    validation_dataset = monai.data.Dataset(
        data=validation_data_list,
        transform=transform
    )

    validation_data_loader = monai.data.DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return [training_data_loader], [validation_data_loader]


def get_test_data_loaders(
    settings
):
    # Get test data list
    test_data_list = read_json_data_file(
        data_file_path=settings['dataset']['data_file'],
        data_dir=settings['dataset']['data_dir'],
        keys='test'
    )[0]

    # Instantiate MONAI-based test data loader
    transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=settings['dataset']['keys']),
            monai.transforms.EnsureChannelFirstd(keys=settings['dataset']['keys'], channel_dim='no_channel'),
            monai.transforms.ToTensord(keys=settings['dataset']['keys'])
        ]
    )

    test_dataset = monai.data.Dataset(
        data=test_data_list,
        transform=transform
    )

    test_data_loader = monai.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return [test_data_loader]


def get_sampling_data_loader(
    settings
):
    # Get test data list
    test_data_list = read_json_data_file(
        data_file_path=settings['dataset']['data_file'],
        data_dir=settings['dataset']['data_dir'],
        keys='training'
    )[0]

    # Instantiate MONAI-based test data loader
    transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=settings['dataset']['keys']),
            monai.transforms.EnsureChannelFirstd(keys=settings['dataset']['keys'], channel_dim='no_channel'),
            monai.transforms.ToTensord(keys=settings['dataset']['keys'])
        ]
    )

    sampling_dataset = monai.data.Dataset(
        data=test_data_list,
        transform=transform
    )

    sampling_data_loader = monai.data.DataLoader(
        dataset=sampling_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return [sampling_data_loader]