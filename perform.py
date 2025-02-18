import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.utils import make_grid
from torchvision import transforms, datasets
import torchvision.models as models

from torchvision.transforms import v2
import copy

from torch.optim import lr_scheduler
import torch.nn.init as init

# Evaluation
from src.model import *
from src.utils.data_load import *
from src.utils.data_prep import *

from collections import Counter



import numpy as np
import random
import time, os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import polars as pl

import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


root_dir = './data'

train_dataset = datasets.GTSRB(root=root_dir, split='train', download=False, transform=transform_train_2)
test_dataset = datasets.GTSRB(root=root_dir, split='test', download=False, transform=transform_test_2)

print(f"Train dataset contains {len(train_dataset)} images")
print(f"Test dataset contains {len(test_dataset)} images")

test_target = list(pl.read_csv('test_target.csv',separator = ',', has_header=False).row(0))

train_target = list(pl.read_csv('train_target.csv',separator = ',', has_header=False).row(0))

class_names = list(pl.read_csv('signnames.csv')['SignName'])

nclasses = len(np.unique(train_target))
all_classes = list(range(nclasses))
classes_per_task = 8
current_classes = []

task = 1
task_classes = all_classes[task * classes_per_task : (task + 1) * classes_per_task]
current_classes.extend(task_classes)
batch_size = 64

train_loader = create_dataloader(train_dataset, train_target, current_classes, batch_size, shuffle = True)
test_loader = create_dataloader(train_dataset, train_target, current_classes, batch_size, shuffle = True)

label_counts = Counter(test_target).most_common()


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self,n_out=10, n_in=1):
        super().__init__()

        # Put the layers here
        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc = nn.Linear(4096, n_out)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x)) ## l'image 1x32x32 devient 32x32x32
        x = F.max_pool2d(x, kernel_size=2, stride=2) ## puis 32x16x16
        x = F.leaky_relu(self.conv2(x)) ## puis devient 64x16x16
        x = F.max_pool2d(x, kernel_size=2, stride=2) ## puis devient 64x8x8
        x = F.leaky_relu(self.conv3(x)) ## pas de changement

        x = x.view(-1,4096) ## 64x8x8 devient 4096

        x = self.fc(x) ## on finit exactement de la même façon

        return x


model = SimpleCNN(n_out=10, n_in=3)
model.to(device)

model = copy.deepcopy(copy_model)
incremental_learning(model, train_dataset, train_target, test_dataset, test_target,
                      num_tasks, classes_per_task, batch_size, num_epochs, lr, device)


