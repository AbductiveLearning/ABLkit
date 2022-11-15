import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

def get_mnist_add():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
    train_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./', train=False, transform=transform), batch_size=1000, shuffle=True)
    
    X = []
    Y = []
    with open('./train_data.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            X.append((train_dataset[int(line[0])][0], train_dataset[int(line[1])][0]))
            Y.append(int(line[2]))
    
    return X, Y, test_loader

if __name__ == "__main__":
    X, Y, test_loader = get_mnist_add()
    print(len(X), len(Y))
    print(X[0][0].shape, X[0][1].shape, Y[0])
