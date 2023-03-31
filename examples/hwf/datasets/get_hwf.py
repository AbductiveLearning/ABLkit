import json
from PIL import Image
from torchvision.transforms import transforms

img_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (1,))
                    ])

def get_data(file, get_pseudo_label):
    X = []
    if get_pseudo_label:
        Z = []
    Y = []
    img_dir = './datasets/data/Handwritten_Math_Symbols/'
    with open(file) as f:
        data = json.load(f)
        for idx in range(len(data)):
            imgs = []
            imgs_pseudo_label = []
            for img_path in data[idx]['img_paths']:
                img = Image.open(img_dir + img_path).convert('L')
                img = img_transform(img)
                imgs.append(img)
                if get_pseudo_label:
                    imgs_pseudo_label.append(img_path.split('/')[0])
            X.append(imgs)
            if get_pseudo_label:
                Z.append(imgs_pseudo_label)
            Y.append(data[idx]['res'])
    
    if get_pseudo_label:
        return X, Z, Y
    else:
        return X, None, Y

def get_hwf(train = True, get_pseudo_label = False):
    if(train):
        file = './datasets/data/expr_train.json'
    else:
        file = './datasets/data/expr_test.json'
    
    return get_data(file, get_pseudo_label)

if __name__ == "__main__":
    train_X, train_Y = get_hwf(train = True)
    test_X, test_Y = get_hwf(train = False)
    print(len(train_X), len(test_X))
    print(len(train_X[0]), train_X[0][0].shape, train_Y[0])
    
