import json
import os
import gdown
import zipfile

from PIL import Image
from torchvision.transforms import transforms

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1,))])

def download_and_unzip(url, zip_file_name):
    try:
        gdown.download(url, zip_file_name)
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(CURRENT_DIR)
        os.remove(zip_file_name)
    except Exception as e:
        if os.path.exists(zip_file_name):
            os.remove(zip_file_name)
        raise Exception(f"An error occurred during download or unzip: {e}. Instead, you can download the dataset from {url} and unzip it in 'examples/hwf/datasets' folder")

def get_dataset(train=True, get_pseudo_label=False):
    data_dir = CURRENT_DIR + '/data'
    
    if not os.path.exists(data_dir):
        print("Dataset not exist, downloading it...")
        url = 'https://drive.google.com/u/0/uc?id=1G07kw-wK-rqbg_85tuB7FNfA49q8lvoy&export=download'
        download_and_unzip(url, os.path.join(CURRENT_DIR, "HWF.zip"))
        print("Download and extraction complete.")
    
    if train:
        file = os.path.join(data_dir, "expr_train.json")
    else:
        file = os.path.join(data_dir, "expr_test.json")

    X = []
    pseudo_label = [] if get_pseudo_label else None
    Y = []
    img_dir = os.path.join(CURRENT_DIR, "data/Handwritten_Math_Symbols/")
    with open(file) as f:
        data = json.load(f)
        for idx in range(len(data)):
            imgs = []
            if get_pseudo_label:
                imgs_pseudo_label = []
            for img_path in data[idx]["img_paths"]:
                img = Image.open(img_dir + img_path).convert("L")
                img = img_transform(img)
                imgs.append(img)
                if get_pseudo_label:
                    label_mappings = {"times": "*", "div": "/"}
                    label = img_path.split("/")[0]
                    label = label_mappings.get(label, label)
                    imgs_pseudo_label.append(label)
            X.append(imgs)
            if get_pseudo_label:
                pseudo_label.append(imgs_pseudo_label)
            Y.append(data[idx]["res"])

    return X, pseudo_label, Y    
