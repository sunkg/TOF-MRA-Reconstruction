#!/usr/bin/env python3

import os
import numpy as np
import h5py
import torch
import random


def center_crop(data, shape):
    if shape[0] <= data.shape[-2]:
        w_from = (data.shape[-2] - shape[0]) // 2
        w_to = w_from + shape[0]
        data = data[..., w_from:w_to, :]
    else:
        w_before = (shape[0] - data.shape[-2]) // 2
        w_after = shape[0] - data.shape[-2] - w_before
        pad = [(0, 0)] * data.ndim
        pad[-2] = (w_before, w_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    if shape[1] <= data.shape[-1]:
        h_from = (data.shape[-1] - shape[1]) // 2
        h_to = h_from + shape[1]
        data = data[..., :, h_from:h_to]
    else:
        h_before = (shape[1] - data.shape[-1]) // 2
        h_after = shape[1] - data.shape[-1] - h_before
        pad = [(0, 0)] * data.ndim
        pad[-1] = (h_before, h_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    return data


class VolumeDataset(torch.utils.data.Dataset):
    def __init__(self, volume, crop=None, mask='Equispaced'):
        super().__init__()
        
        self.volume = volume
        self.crop = crop
        h5 = h5py.File(volume, 'r')
        data = np.array(h5['recon_cc']).astype(np.complex64)
        Slab, Slice, Coil, H, W = data.shape
        self.data = data.reshape(Slab*Slice, Coil, H, W)
        
        # Read the slices without overlap between neighboring slabs
        select = np.arange(0,39).tolist() + np.arange(51,81).tolist() + np.arange(94, 126).tolist() + np.arange(139, 169).tolist() + np.arange(181, 218).tolist()
        if self.data.shape[0] > select[-1]:
            self.data = self.data[select] # 168

        self.length, self.channels = self.data.shape[0:2]
        h5.close()

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        item = self.data[index][()]
        item = item / np.abs(item).max() 

        if (self.crop is not None) & (len(self.crop) == 2): 
            item = center_crop(item, (self.crop[0], self.crop[1]))

        return item


class DummyVolumeDataset(torch.utils.data.Dataset):
    def __init__(self, ref):
        super().__init__()
        sample = ref[0]
        self.shape = sample.shape
        self.dtype = sample.dtype
        self.len = len(ref)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return np.zeros(self.shape, dtype=self.dtype)


class AlignedVolumesDataset(torch.utils.data.Dataset):
    def __init__(self, *volumes, crop=None, mask='Equispaced'):
        super().__init__()
        volumes = [VolumeDataset(x, crop, mask=mask) for x in volumes]
        
        self.volumes = volumes

    def __len__(self):
        return len(self.volumes[0])

    def __getitem__(self, index):
        images = [volume[index] for volume in self.volumes]
        return images


def get_train_volume_datasets(basepath, crop=None, mask='Equispaced'):
    datasets = []
    files = os.listdir(basepath)
    random.shuffle(files)
    for file in files:
        dataset = [os.path.join(basepath, file, 'recon_cc.hdf5')] # Read data
        dataset = AlignedVolumesDataset(*dataset, crop=crop, mask=mask)
        datasets.append(dataset)
    return datasets 

def get_eval_volume_datasets(basepath, crop=None, mask='Equispaced'):
    datasets = []
    files = os.listdir(basepath)
    files.sort()
    for file in files:
        dataset = [os.path.join(basepath, file, 'recon_cc.hdf5')]
        dataset = AlignedVolumesDataset(*dataset, crop=crop, mask=mask)
        datasets.append(dataset)
    return datasets

