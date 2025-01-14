import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn

""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    torch.manual_seed(42)
    
    """ Directory for storing files """
    create_dir("remove_bg")
    
    """ Loading model """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load("model.pth", map_location=device)
    model.eval()
    
    """ Load the dataset """
    data_x = glob("images/*")
    
    with torch.no_grad():
        for path in tqdm(data_x, total=len(data_x)):
            """ Extracting name """
            name = path.split("/")[-1].split(".")[0]
            
            """ Read the image """
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            h, w, _ = image.shape
            x = cv2.resize(image, (W, H))
            x = x/255.0
            x = x.astype(np.float32)
            x = np.transpose(x, (2, 0, 1))  # HWC to CHW format
            x = torch.from_numpy(x).unsqueeze(0).to(device)
            
            """ Prediction """
            y = model(x)[0]
            y = y.cpu().numpy()
            y = np.transpose(y, (1, 2, 0))  # CHW to HWC format
            y = cv2.resize(y, (w, h))
            y = np.expand_dims(y, axis=-1)
            y = y > 0.5
            
            photo_mask = y
            background_mask = np.abs(1-y)
            
            masked_photo = image * photo_mask
            background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1)
            background_mask = background_mask * [0, 0, 255]
            final_photo = masked_photo + background_mask
            
            cv2.imwrite(f"remove_bg/{name}.png", final_photo)