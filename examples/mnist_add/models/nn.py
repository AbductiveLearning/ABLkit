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


import numpy as np
import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, image_size=(28, 28)):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 3), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(nn.Conv2d(16, 16, 3), nn.ReLU())

        feature_map_size = (np.array(image_size) // 2 - 2) // 2 - 2
        num_features = 16 * feature_map_size[0] * feature_map_size[1]

        self.fc1 = nn.Sequential(nn.Linear(num_features, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


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
        self.fc3 = nn.Sequential(nn.Linear(84, num_classes))

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
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Sequential(nn.Linear(num_classes, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, image_size[0] * image_size[1]), nn.ReLU())

    def forward(self, x):
        x = self.base_model(x)
        # x = self.softmax(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
