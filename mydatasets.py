import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
# step1: 定义MyDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__

class MyDatasets(Dataset):
    def __init__(self, root_dir, names_file, transform=None,drop_last=False):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.root_dir +"/"+self.names_list[idx].split(' ')[0]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = Image.open(image_path).convert('RGB')   # use skitimage
        image = image.resize((224, 224), Image.BILINEAR)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'image': image, 'label': label}
        pos = torch.ones(4).cuda()
        if self.transform:
            image = self.transform(image)
        return image,label,pos
