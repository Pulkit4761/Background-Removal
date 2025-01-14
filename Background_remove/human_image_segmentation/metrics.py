import torch
import torch.nn as nn
import torch.nn.functional as F

def iou(y_true, y_pred):
    #Calculate Intersection over Union (IoU)

    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    x = (intersection + 1e-15) / (union + 1e-15)
    return x

class DiceCoef(nn.Module):
#Calculate Dice Coefficient

    def __init__(self, smooth=1e-15):
        super(DiceCoef, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        intersection = torch.sum(y_true * y_pred)
        return (2. * intersection + self.smooth) / (torch.sum(y_true) + torch.sum(y_pred) + self.smooth)

class DiceLoss(nn.Module):
#Calculate Dice Loss

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.dice_coef = DiceCoef()

    def forward(self, y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)

