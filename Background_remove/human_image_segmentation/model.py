import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SqueezeAndExcite(nn.Module):
    def __init__(self, channel, ratio=8):
        super(SqueezeAndExcite, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ASPP(nn.Module):
    def __init__(self, in_channels):
        super(ASPP, self).__init__()
        
        # Image Pooling
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Dilated convolutions
        self.conv_dilated_6 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv_dilated_12 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv_dilated_18 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 conv
        self.conv_final = nn.Sequential(
            nn.Conv2d(256*5, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        size = x.size()
        pool = F.interpolate(self.pool(x), size[2:], mode='bilinear', align_corners=True)
        conv1 = self.conv1(x)
        conv_dilated_6 = self.conv_dilated_6(x)
        conv_dilated_12 = self.conv_dilated_12(x)
        conv_dilated_18 = self.conv_dilated_18(x)
        
        concat = torch.cat([pool, conv1, conv_dilated_6, conv_dilated_12, conv_dilated_18], dim=1)
        return self.conv_final(concat)

class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super(DeepLabV3Plus, self).__init__()
        
        # ResNet50 backbone
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-3])
        
        # Get intermediate layers
        self.low_level_features = nn.Sequential(*list(resnet.children())[:5])
        
        # ASPP
        self.aspp = ASPP(1024)  # ResNet50 conv4 output channels
        
        # Low-level features conv
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Final convolutions
        self.se1 = SqueezeAndExcite(304)  # 256 + 48 channels
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.se2 = SqueezeAndExcite(256)
        
        self.final = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        low_level_feat = self.low_level_features(x)
        encoder_out = self.encoder(x)
        
        # ASPP
        aspp_out = self.aspp(encoder_out)
        aspp_out = F.interpolate(aspp_out, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        # Low-level features
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # Concatenate ASPP and low-level features
        x = torch.cat([aspp_out, low_level_feat], dim=1)
        
        # Apply SE and final convolutions
        x = self.se1(x)
        x = self.final_conv(x)
        x = self.se2(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return self.final(x)

if __name__ == "__main__":
    # Example usage
    model = DeepLabV3Plus()
    model.eval()
    
    # Test with random input
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")