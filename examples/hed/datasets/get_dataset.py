import os
import os.path as osp
import pickle
import random
import zipfile
from collections import defaultdict
from PIL import Image

import gdown
import numpy as np
from torchvision.transforms import transforms

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def download_and_unzip(url, zip_file_name):
    try:
        gdown.download(url, zip_file_name)
        with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
            zip_ref.extractall(CURRENT_DIR)
        os.remove(zip_file_name)
    except Exception as e:
        if os.path.exists(zip_file_name):
            os.remove(zip_file_name)
        raise Exception(
            f"An error occurred during download or unzip: {e}. Instead, you can download "
            + f"the dataset from {url} and unzip it in 'examples/hed/datasets' folder"
        )


def get_pretrain_data(labels, image_size=(28, 28, 1)):
    transform = transforms.Compose([transforms.ToTensor()])
    X = []
    img_dir = osp.join(CURRENT_DIR, "mnist_images")
    for label in labels:
        label_path = osp.join(img_dir, label)
        img_path_list = os.listdir(label_path)
        for img_path in img_path_list:
            with Image.open(osp.join(label_path, img_path)) as img:
                img = img.convert("L")
                img = img.resize((image_size[1], image_size[0]))
                img_array = np.array(img, dtype=np.float32)
                normalized_img = (img_array - 127) / 128.0
                X.append(normalized_img)

    Y = [img.copy().reshape(image_size[0] * image_size[1] * image_size[2]) for img in X]
    X = [transform(img[:, :, np.newaxis]) for img in X]
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


def get_dataset(dataset="mnist", train=True):
    data_dir = CURRENT_DIR + "/mnist_images"

    if not os.path.exists(data_dir):
        print("Dataset not exist, downloading it...")
        url = "https://drive.google.com/u/0/uc?id=1XoJDjO3cNUdytqVgXUKOBe9dOcUBobom&export=download"
        download_and_unzip(url, os.path.join(CURRENT_DIR, "HED.zip"))
        print("Download and extraction complete.")

    if train:
        file = os.path.join(data_dir, "expr_train.json")
    else:
        file = os.path.join(data_dir, "expr_test.json")

    if dataset == "mnist":
        file = osp.join(CURRENT_DIR, "mnist_equation_data_train_len_26_test_len_26_sys_2_.pk")
    elif dataset == "random":
        file = osp.join(CURRENT_DIR, "random_equation_data_train_len_26_test_len_26_sys_2_.pk")
    else:
        raise ValueError("Undefined dataset")

    with open(file, "rb") as f:
        img_dataset = pickle.load(f)

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

    equations_by_len = divide_equations_by_len(X, Y)
    return equations_by_len
