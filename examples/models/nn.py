# coding: utf-8
# ================================================================#
#   Copyright (C) 2021 Freecss All rights reserved.
#
#   File Name     ：lenet5.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2021/03/03
#   Description   ：
#
# ================================================================#


import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, image_size=(28, 28)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)

        feature_map_size = (np.array(image_size) // 2 - 2) // 2 - 2
        num_features = 16 * feature_map_size[0] * feature_map_size[1]

        self.fc1 = nn.Linear(num_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """前向传播函数"""
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SymbolNet(nn.Module):
    def __init__(self, num_classes=4, image_size=(28, 28, 1)):
        super(SymbolNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32, momentum=0.99, eps=0.001),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64, momentum=0.99, eps=0.001),
        )

        num_features = 64 * (image_size[0] // 4 - 1) * (image_size[1] // 4 - 1)
        self.fc1 = nn.Sequential(nn.Linear(num_features, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(84, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class SymbolNetAutoencoder(nn.Module):
    def __init__(self, num_classes=4, image_size=(28, 28, 1)):
        super(SymbolNetAutoencoder, self).__init__()
        self.base_model = SymbolNet(num_classes, image_size)
        self.fc1 = nn.Sequential(nn.Linear(num_classes, 100), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(100, image_size[0] * image_size[1]), nn.ReLU()
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
