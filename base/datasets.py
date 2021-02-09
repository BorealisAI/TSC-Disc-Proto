# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved. #
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from base.utils import z_norm_seqs, np2tensor


def zero_start(label_np):
    """
    Guarantees label starts from 0.
    """
    min_val = np.amin(label_np)
    assert min_val >= 0, min_val
    return label_np - min_val


def load_raw_data(data_file):
    """
    Load txt to numpy, split data and labels.
    """
    assert os.path.isfile(data_file), "{} not exists.".format(data_file)
    full_np = np.loadtxt(data_file)
    data_np, label_np = full_np[:, 1:], full_np[:, 0].astype(int)
    assert np.array_equal(label_np, full_np[:, 0])
    return data_np, zero_start(label_np)


class UCR_dataset(Dataset):
    def __init__(self, data_np, label_np, Class_N, data_transform, z_norm=True):
        if z_norm:
            self.data_mat = z_norm_seqs(data_np)
            assert self.data_mat.shape == data_np.shape, (self.data_mat.shape, data_np.shape)
        else:
            self.data_mat = data_np
        self.label_list = label_np.tolist()
        self.set_n = len(self.label_list)
        assert len(self.data_mat.shape) == 2, self.data_mat.shape
        assert self.data_mat.shape[0] == self.set_n, self.data_mat.shape

        assert Class_N > 0, Class_N
        self.cls_n = Class_N

        self.data_transform = data_transform

    def __len__(self):
        return self.set_n

    def __getitem__(self, index):
        np_arr = self.data_mat[index]
        if self.data_transform:
            data = self.data_transform(np_arr)
        else:
            data = np_arr

        label = self.label_list[index]
        assert isinstance(label, int), type(label)
        assert 0 <= label < self.cls_n, label

        sample = {
            "data": data,
            "label": label,
        }
        return sample


def build_loader(raw_data_file, Class_N, batch_size, shuffle, drop_last=False, num_workers=4, pin_mem=True):
    data_np, label_np = load_raw_data(raw_data_file)

    data_transform = transforms.Compose([np2tensor()])

    dataset = UCR_dataset(data_np, label_np, Class_N, data_transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    return dataloader, len(dataset)
