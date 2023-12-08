import os
import os.path as osp
import pickle
import random
from collections import defaultdict

import cv2
import numpy as np
from torchvision.transforms import transforms

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_data(img_dataset, train):
    X, Y = [], []
    if train:
        positive = img_dataset["train:positive"]
        negative = img_dataset["train:negative"]
    else:
        positive = img_dataset["test:positive"]
        negative = img_dataset["test:negative"]

    for equation in positive:
        equation = equation.astype(np.float32)
        img_list = np.vsplit(equation, equation.shape[0])
        X.append(img_list)
        Y.append(1)

    for equation in negative:
        equation = equation.astype(np.float32)
        img_list = np.vsplit(equation, equation.shape[0])
        X.append(img_list)
        Y.append(0)

    return X, None, Y


def get_pretrain_data(labels, image_size=(28, 28, 1)):
    transform = transforms.Compose([transforms.ToTensor()])
    X = []
    img_dir = osp.join(CURRENT_DIR, "mnist_images")
    for label in labels:
        label_path = osp.join(img_dir, label)
        img_path_list = os.listdir(label_path)
        for img_path in img_path_list:
            img = cv2.imread(osp.join(label_path, img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (image_size[1], image_size[0]))
            X.append(np.array(img, dtype=np.float32))

    X = [((img[:, :, np.newaxis] - 127) / 128.0) for img in X]
    Y = [img.copy().reshape(image_size[0] * image_size[1] * image_size[2]) for img in X]

    X = [transform(img) for img in X]
    return X, Y


def divide_equations_by_len(equations, labels):
    equations_by_len = {1: defaultdict(list), 0: defaultdict(list)}
    for i, equation in enumerate(equations):
        equations_by_len[labels[i]][len(equation)].append(equation)
    return equations_by_len


def split_equation(equations_by_len, prop_train, prop_val):
    """
    Split the equations in each length to training and validation data according to the proportion
    """
    train_equations_by_len = {1: dict(), 0: dict()}
    val_equations_by_len = {1: dict(), 0: dict()}

    for label in range(2):
        for equation_len, equations in equations_by_len[label].items():
            random.shuffle(equations)
            train_equations_by_len[label][equation_len] = equations[
                : len(equations) // (prop_train + prop_val) * prop_train
            ]
            val_equations_by_len[label][equation_len] = equations[
                len(equations) // (prop_train + prop_val) * prop_train :
            ]

    return train_equations_by_len, val_equations_by_len


def get_hed(dataset="mnist", train=True):
    if dataset == "mnist":
        file = osp.join(CURRENT_DIR, "mnist_equation_data_train_len_26_test_len_26_sys_2_.pk")
    elif dataset == "random":
        file = osp.join(CURRENT_DIR, "random_equation_data_train_len_26_test_len_26_sys_2_.pk")
    else:
        raise ValueError("Undefined dataset")

    with open(file, "rb") as f:
        img_dataset = pickle.load(f)
    X, _, Y = get_data(img_dataset, train)
    equations_by_len = divide_equations_by_len(X, Y)

    return equations_by_len


if __name__ == "__main__":
    get_hed()
