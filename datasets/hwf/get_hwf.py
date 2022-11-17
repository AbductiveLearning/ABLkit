import json
from PIL import Image
from torchvision.transforms import transforms

img_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (1,))
                    ])

def get_data(file, precision_num = 2):
    X = []
    Y = []
    img_dir = './datasets/hwf/data/Handwritten_Math_Symbols/'
    with open(file) as f:
        data = json.load(f)
        for idx in range(len(data)):
            imgs = []
            for img_path in data[idx]['img_paths']:
                img = Image.open(img_dir + img_path).convert('L')
                img = img_transform(img)
                imgs.append(img)
            X.append(imgs)
            Y.append(round(data[idx]['res'], precision_num))
    return X, Y

def get_hwf(precision_num = 2):
    train_X, train_Y = get_data('./datasets/hwf/data/expr_train.json', precision_num)
    test_X, test_Y = get_data('./datasets/hwf/data/expr_test.json', precision_num)
    
    return train_X, train_Y, test_X, test_Y

if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y = get_hwf()
    print(len(train_X), len(test_X))
    print(len(train_X[0]), train_X[0][0].shape, train_Y[0])
    
