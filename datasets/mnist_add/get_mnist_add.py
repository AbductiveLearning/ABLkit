import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class MNIST_Addition(Dataset):
    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, l = self.data[index]
        return self.dataset[i1][0], self.dataset[i2][0], l

def get_mnist_add():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
    train_dataset = MNIST_Addition(torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform), './train_data.txt')
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./', train=False, transform=transform), batch_size=1000, shuffle=True)
    X = []
    Y = []
    for i1, i2, l in train_dataset:
        X.append([i1, i2])
        Y.append(l)
    return X, Y, test_loader

if __name__ == "__main__":
    X, Y, test_loader = get_mnist_add()
    print(len(X), len(Y))
    print(X[0][0].shape, X[0][1].shape, Y[0])
    