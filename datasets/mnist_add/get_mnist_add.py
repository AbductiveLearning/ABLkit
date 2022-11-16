import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

def get_data(file, img_dataset):
    X = []
    Y = []
    with open(file) as f:
        for line in f:
            line = line.strip().split(' ')
            X.append([img_dataset[int(line[0])][0], img_dataset[int(line[1])][0]])
            Y.append(int(line[2]))
    return X, Y

def get_mnist_add():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
    img_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
    
    train_X, train_Y = get_data('./train_data.txt', img_dataset)
    test_X, test_Y = get_data('./test_data.txt', img_dataset)
    
    return train_X, train_Y, test_X, test_Y

if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y = get_mnist_add()
    print(len(train_X), len(test_X))
    print(train_X[0][0].shape, train_X[0][1].shape, train_Y[0])
    
