import os

import torchvision
from torchvision.transforms import transforms

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_dataset(train=True, get_pseudo_label=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    img_dataset = torchvision.datasets.MNIST(
        root=CURRENT_DIR, train=train, download=True, transform=transform
    )
    if train:
        file = os.path.join(CURRENT_DIR, "train_data.txt")
    else:
        file = os.path.join(CURRENT_DIR, "test_data.txt")

    X = []
    pseudo_label = [] if get_pseudo_label else None
    Y = []
    with open(file) as f:
        for line in f:
            x1, x2, y = map(int, line.strip().split(" "))
            X.append([img_dataset[x1][0], img_dataset[x2][0]])
            if get_pseudo_label:
                pseudo_label.append([img_dataset[x1][1], img_dataset[x2][1]])
            Y.append(y)
    return X, pseudo_label, Y
