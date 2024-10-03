# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
import random

identity = lambda x: x


#%% Customize based on dataset

class SimpleDataset:
    def __init__(self, data_path, data_file_list, transform, target_transform=identity):
        label = []
        data = []
        k = 0
        data_dir_list = data_file_list.replace(" ","").split(',')
        for data_file in data_dir_list:
            img_dir = data_path + '/' + data_file
            for i in os.listdir(img_dir):
                file_dir = os.path.join(img_dir, i)
                for j in os.listdir(file_dir):
                    data.append(file_dir + '/' + j)
                    label.append(k)
                k += 1
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform
        self.checkimgsize(self.data[random.randint(0, len(self.label) - 1 )])
    
    def checkimgsize(self, data):
        data_path = os.path.join(data)
        data = Image.open(data_path).convert('RGB')
        if data.size == (84, 84):
            raise RuntimeError("Please use raw images instead of fixed resolution(84, 84) images !")

    def __getitem__(self, i):
        image_path = os.path.join(self.data[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.label[i] - min(self.label))
        return img, target

    def __len__(self):
        return len(self.label)



class SetDataset:
    def __init__(self, data_path, data_file_list, batch_size, transform):
        label = []
        data = []
        k = 0
        data_dir_list = data_file_list.replace(" ","").split(',')
        for data_file in data_dir_list:
            img_dir = data_path + '/' + data_file
            for i in os.listdir(img_dir):
                file_dir = os.path.join(img_dir, i)
                for j in os.listdir(file_dir):
                    data.append(file_dir + '/' + j)
                    label.append(k)
                k += 1
        self.data = data
        self.label = label
        self.transform = transform
        self.cl_list = np.unique(self.label).tolist()
        self.checkimgsize(self.data[random.randint(0, len(self.label) - 1 )])

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.data, self.label):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def checkimgsize(self, data):
        data_path = os.path.join(data)
        data = Image.open(data_path).convert('RGB')
        if data.size == (84, 84):
            raise RuntimeError("Please use raw images instead of fixed resolution(84, 84) images !")

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)




class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

import torch.distributed as dist
from torch.utils.data.sampler import Sampler

class DistributedEpisodicBatchSampler(Sampler):
    def __init__(self, n_classes, n_way, n_episodes, rank=None, world_size=None):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.rank = rank if rank is not None else dist.get_rank()
        self.world_size = world_size if world_size is not None else dist.get_world_size()

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            # Ensure different processes generate the same random indices
            torch.manual_seed(i * self.world_size + self.rank)
            yield torch.randperm(self.n_classes)[:self.n_way]



