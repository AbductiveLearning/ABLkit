import os
import numpy as np

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_dataset(fname, get_pseudo_label=True):
    fname = os.path.join(CURRENT_DIR, fname)
    data = np.load(fname)
    X = data["X"]
    X = [[emb.astype(np.float32)] for emb in X]
    pseudo_label = data["pseudo_label"].astype(int).tolist() if get_pseudo_label else None
    Y = data["Y"][:, :4].astype(int).tolist()
    Y = [tuple(y) for y in Y]
    return X, pseudo_label, Y

if __name__ == '__main__':
    dataset = get_dataset("val.npz")