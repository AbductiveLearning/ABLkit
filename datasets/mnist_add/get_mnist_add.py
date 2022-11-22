import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

def get_data(file, img_dataset, get_pseudo_label):
    X = []
    if get_pseudo_label:
        Z = []
    Y = []
    with open(file) as f:
        for line in f:
            line = line.strip().split(' ')
            X.append([img_dataset[int(line[0])][0], img_dataset[int(line[1])][0]])
            if get_pseudo_label:
                Z.append([img_dataset[int(line[0])][1], img_dataset[int(line[1])][1]])
            Y.append(int(line[2]))
            
    if get_pseudo_label:
        return X, Z, Y
    else:
        return X, None, Y

def get_mnist_add(train = True, get_pseudo_label = False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
    img_dataset = torchvision.datasets.MNIST(root='./datasets/mnist_add/', train=train, download=True, transform=transform)
    
    if train:
        file = './datasets/mnist_add/train_data.txt'
    else:
        file = './datasets/mnist_add/test_data.txt'
    
    return get_data(file, img_dataset, get_pseudo_label)
    

if __name__ == "__main__":
    train_X, train_Y = get_mnist_add(train = True)
    test_X, test_Y = get_mnist_add(train = False)
    print(len(train_X), len(test_X))
    print(train_X[0][0].shape, train_X[0][1].shape, train_Y[0])
    
