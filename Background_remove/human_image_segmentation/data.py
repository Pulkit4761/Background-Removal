import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    train_x, test_x = train_test_split(X, test_size=split, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split, random_state=42)

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            x2 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            y2 = y

            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = CoarseDropout(p=1, min_holes=3, max_holes=10, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4, y5]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            try:
                aug = CenterCrop(H, W, p=1.0)
                augmented = aug(image=i, mask=m)
                i = augmented["image"]
                m = augmented["mask"]

            except Exception as e:
                i = cv2.resize(i, (W, H))
                m = cv2.resize(m, (W, H))

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    np.random.seed(42)

    data_path = "segmentation"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train:\t {len(train_x)} - {len(train_y)}")
    print(f"Test:\t {len(test_x)} - {len(test_y)}")

    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)