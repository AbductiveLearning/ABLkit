import json
from PIL import Image
from torchvision.transforms import transforms

img_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (1,))
                    ])

def get_data(file):
    X = []
    Y = []
    img_dir = './data/Handwritten_Math_Symbols/'
    with open(file) as f:
        data = json.load(f)
        for idx in range(len(data)):
            imgs = []
            for img_path in data[idx]['img_paths']:
                img = Image.open(img_dir + img_path).convert('L')
                img = img_transform(img)
                imgs.append(img)
            X.append(imgs)
            Y.append(data[idx]['res'])
    return X, Y

def get_hwf():
    train_X, train_Y = get_data('./data/expr_train.json')
    test_X, test_Y = get_data('./data/expr_test.json')
    
    return train_X, train_Y, test_X, test_Y

if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y = get_hwf()
    print(len(train_X), len(test_X))
    print(len(train_X[0]), train_X[0][0].shape, train_Y[0])
    
