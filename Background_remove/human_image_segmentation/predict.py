import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from metrics import DiceLoss, DiceCoef, iou

H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    create_dir("test_images/mask")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(r"C:\Users\91767\Desktop\GDG_project\Background_remove\human_image_segmentation\model.py", map_location=device)
    model.eval() 
 
    data_x = glob(r"Background_remove/images/people_segmentation/images")
    
    with torch.no_grad():  
        for path in tqdm(data_x, total=len(data_x)):
            name = path.split("/")[-1].split(".")[0]
            
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            h, w, _ = image.shape
            x = cv2.resize(image, (W, H))
            x = x/255.0
            x = x.astype(np.float32)
            
            x = np.transpose(x, (2, 0, 1)) 
            x = torch.from_numpy(x).unsqueeze(0)  
            x = x.to(device)
            
            y = model(x)[0]  
            y = y.cpu().numpy()  
            if len(y.shape) == 3:  
                y = np.transpose(y, (1, 2, 0))
            
            y = cv2.resize(y, (w, h))
            y = np.expand_dims(y, axis=-1)
            
            masked_image = image * y
            line = np.ones((h, 10, 3)) * 128
            cat_images = np.concatenate([image, line, masked_image], axis=1)
            cv2.imwrite(f"test_images/mask/{name}.png", cat_images)