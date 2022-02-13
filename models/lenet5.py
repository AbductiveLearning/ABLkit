# coding: utf-8
#================================================================#
#   Copyright (C) 2021 Freecss All rights reserved.
#   
#   File Name     ：lenet5.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2021/03/03
#   Description   ：
#
#================================================================#

import sys
sys.path.append("..")

import torchvision

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from models.basic_model import BasicModel

import utils.plog as plog

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)

        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 13)

    def forward(self, x):
        '''前向传播函数'''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        #x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

class Params:
    imgH = 28
    imgW = 28
    keep_ratio = True
    saveInterval = 10
    batchSize = 16
    num_workers = 16

def get_data():  #数据预处理
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])
                                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #训练集
    train_set = torchvision.datasets.MNIST(root='data/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1024, shuffle=True, num_workers = 16)
    #测试集
    test_set = torchvision.datasets.MNIST(root='data/', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1024, shuffle = False, num_workers = 16)
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    return train_loader, test_loader, classes

if __name__ == "__main__":
    recorder = plog.ResultRecorder()
    cls = LeNet5()
    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BasicModel(cls, criterion, optimizer, None, device, Params(), recorder)
    
    train_loader, test_loader, classes = get_data()

    #model.val(test_loader, print_prefix = "before training")
    model.fit(train_loader, n_epoch = 100)
    model.val(test_loader, print_prefix = "after trained")
    res = model.predict(test_loader, print_prefix = "predict")
    print(res.argmax(axis=1)[:10])

