import os
import json

from PIL import Image
from torchvision.transforms import transforms

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

img_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1,))]
)


def get_data(file, get_pseudo_label):
    X, Y = [], []
    if get_pseudo_label:
        Z = []
    img_dir = os.path.join(CURRENT_DIR, "data/Handwritten_Math_Symbols/")
    with open(file) as f:
        data = json.load(f)
        for idx in range(len(data)):
            imgs = []
            imgs_pseudo_label = []
            for img_path in data[idx]["img_paths"]:
                img = Image.open(img_dir + img_path).convert("L")
                img = img_transform(img)
                imgs.append(img)
                if get_pseudo_label:
                    imgs_pseudo_label.append(img_path.split("/")[0])
            X.append(imgs)
            if get_pseudo_label:
                Z.append(imgs_pseudo_label)
            Y.append(data[idx]["res"])

    if get_pseudo_label:
        return X, Z, Y
    else:
        return X, None, Y


def get_hwf(train=True, get_gt_pseudo_label=False):
    if train:
        file = os.path.join(CURRENT_DIR, "data/expr_train.json")
    else:
        file = os.path.join(CURRENT_DIR, "data/expr_test.json")

    return get_data(file, get_gt_pseudo_label)
