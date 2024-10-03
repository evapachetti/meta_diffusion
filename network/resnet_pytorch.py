from torchvision import models
import torch
import torch.nn as nn

class Resnet18(nn.Module):
    def __init__(self, params, dataset='picai'):
        super(Resnet18, self).__init__()

        self.n_shot = params.n_shot
        self.n_query = params.n_query
        self.n_add = params.n_add
        self.n_way = getattr(params, 'train_n_way', params.test_n_way)
        self.image_size = params.image_size
        self.dataset = dataset

        model = models.resnet18(pretrained=True)
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()  # Cancel classification layer to get only feature extractor

        self.features = model
        self.flatten = nn.Flatten()
        self.feat_dim = [512, 8, 8]

    def forward(self, x):
        x = self.features(x)
        if self.image_size == 128:
            out = torch.reshape(x, ((self.n_shot + self.n_add + self.n_query) * self.n_way, 512, 4, 4))
        elif self.image_size == 224:
            out = torch.reshape(x, ((self.n_shot + self.n_add + self.n_query) * self.n_way, 512, 7, 7))
        return out
