import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Recall, Precision
import pandas as pd
from tqdm import tqdm
from model import DeepLabV3Plus  # Make sure this is your PyTorch model
from metrics import DiceLoss, DiceCoef, iou

""" Global parameters """
H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*png")))
    y = sorted(glob(os.path.join(path, "mask", "*png")))
    return x, y

class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]

        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image/255.0
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension

        # Convert to tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0
    
    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        with torch.no_grad():
            epoch_dice += DiceCoef(masks, torch.sigmoid(outputs)).item()
            epoch_iou += iou(masks, torch.sigmoid(outputs)).item()

    return epoch_loss/len(loader), epoch_dice/len(loader), epoch_iou/len(loader)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            epoch_loss += loss.item()
            epoch_dice += DiceCoef(masks, torch.sigmoid(outputs)).item()
            epoch_iou += iou(masks, torch.sigmoid(outputs)).item()

    return epoch_loss/len(loader), epoch_dice/len(loader), epoch_iou/len(loader)

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    create_dir("files")
    
    batch_size = 2
    lr = 1e-4
    num_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = "new_data"
    train_path = os.path.join(r"C:\Users\91767\Desktop\GDG_project\Background_remove\images\people_segmentation\segmentation\train.txt")
    valid_path = os.path.join(r"C:\Users\91767\Desktop\GDG_project\Background_remove\images\people_segmentation\segmentation\trainval.txt")
    
    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    
    train_dataset = SegmentationDataset(train_x, train_y)
    valid_dataset = SegmentationDataset(valid_x, valid_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    model = DeepLabV3Plus().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7, verbose=True)
    criterion = DiceLoss()
 
    best_valid_loss = float('inf')
    csv_data = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        valid_loss, valid_dice, valid_iou = validate_one_epoch(
            model, valid_loader, criterion, device
        )
        
        scheduler.step(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join("files", "model.pth"))
            print("Saved best model!")
        
        row = {
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'train_iou': train_iou,
            'valid_loss': valid_loss,
            'valid_dice': valid_dice,
            'valid_iou': valid_iou,
            'lr': optimizer.param_groups[0]['lr']
        }
        csv_data.append(row)
        pd.DataFrame(csv_data).to_csv(os.path.join("files", "data.csv"), index=False)
        
        print(f'Train Loss: {train_loss:.4f} - Dice: {train_dice:.4f} - IoU: {train_iou:.4f}')
        print(f'Valid Loss: {valid_loss:.4f} - Dice: {valid_dice:.4f} - IoU: {valid_iou:.4f}')