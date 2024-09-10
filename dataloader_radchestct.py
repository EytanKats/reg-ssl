import os
import json
import monai
import random
import nibabel as nib

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, WeightedRandomSampler

from convex_adam_utils import MINDSSC


KEYS = ['image_1', 'image_2', 'seg_1', 'seg_2']


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


class GPUCacheDataset(IterableDataset):
    def __init__(self, data_list, batch_size):
        self.data_list = data_list
        self.batch_size = batch_size

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
                self.images_cache[data_pair['image_1']] = torch.tensor(nib.load(data_pair['image_1']).get_fdata(), dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()
                self.mind_cache[data_pair['image_1']] = F.avg_pool3d(MINDSSC(self.images_cache[data_pair['image_1']].cuda(), 1, 2), 2).cpu()

            if data_pair['image_2'] not in self.images_cache:
                self.images_cache[data_pair['image_2']] = torch.tensor(nib.load(data_pair['image_2']).get_fdata(), dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()
                self.mind_cache[data_pair['image_2']] = F.avg_pool3d(MINDSSC(self.images_cache[data_pair['image_2']].cuda(), 1, 2), 2).cpu()

        self.weights = torch.ones(len(data_list))

    def generate(self):
        while True:
            idxs = torch.tensor(list(WeightedRandomSampler(self.weights, self.batch_size, replacement=True))).long()
            out = {
                'image_1': torch.concatenate([self.images_cache[self.samples[idx]['image_1']] for idx in idxs], dim=0),
                'image_2':  torch.concatenate([self.images_cache[self.samples[idx]['image_2']] for idx in idxs], dim=0),
                'mind_1': torch.concatenate([self.mind_cache[self.samples[idx]['image_1']] for idx in idxs], dim=0),
                'mind_2': torch.concatenate([self.mind_cache[self.samples[idx]['image_2']] for idx in idxs], dim=0),
                'idx': torch.tensor([self.samples[idx]['idx'] for idx in idxs])
            }

            yield out

    def __iter__(self):
        return iter(self.generate())


def get_data_loader(
        data_file,
        root_dir,
        key='training',
        batch_size=1,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        fast=False,
        max_samples_num=None,
        random_samples=False,
):

    data_list = read_json_data_file(data_file_path=data_file, data_dir=root_dir, keys=[key])[0]

    if max_samples_num is not None and random_samples:
        indices = random.sample(range(0, len(data_list) - 1), max_samples_num)
        data_list = [data_list[idx] for idx in indices]
    elif max_samples_num is not None:
        data_list = data_list[:max_samples_num]

    if fast:
        data_loader = GPUCacheDataset(data_list=data_list, batch_size=batch_size)
    else:

        radchestct_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=KEYS),
                monai.transforms.EnsureChannelFirstd(keys=KEYS, channel_dim='no_channel'),
                monai.transforms.ToTensord(keys=KEYS)
            ]
        )

        ds = monai.data.Dataset(data=data_list, transform=radchestct_transform)
        data_loader = monai.data.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True
        )

    return data_loader
