# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler, DistributedEpisodicBatchSampler
from abc import abstractmethod


def range_transform(image):
    image = image.type(torch.float32)
    return (torch.rand(image.shape, dtype=torch.float32) + image * 255.) / 256.

class TransformLoader:
    def __init__(self, image_size, range_transform):        
        self.image_size = image_size
        self.range_transform = range_transform

    def get_composed_transform(self):
            transform = transforms.Compose([transforms.ToTensor(), self.range_transform])
            return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, csv_file, image_size, batch_size, json_read=False):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.json_read = json_read


    def get_data_loader(self, csv_file,data):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform()
        dataset = SimpleDataset(csv_file, transform)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, csv_path, image_size, n_way, n_support, n_query, n_episode):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.csv_path = csv_path
        self.trans_loader = TransformLoader(image_size, range_transform)

    def get_data_loader(self, data, rank, world_size):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform()
        dataset = SetDataset(self.csv_path, self.batch_size, transform)
        #sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        ddp_sampler = DistributedEpisodicBatchSampler(len(dataset), self.n_way, self.n_episode,world_size, rank)
        data_loader_params = dict(batch_sampler=ddp_sampler, num_workers=0, pin_memory=True, shuffle=False)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader



