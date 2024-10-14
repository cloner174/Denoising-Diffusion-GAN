import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import random



class CombinedDataset(Dataset):
    def __init__(self, patches, labels, train=True):
        self.patches = patches
        self.labels = labels
        self.train = False
    def __len__(self):
        return len(self.patches)
    def augment_patch(self, patch):
        if random.random() > 0.5:
            patch = np.flip(patch, axis=1)
        if random.random() > 0.5:
            patch = np.flip(patch, axis=2)
        if random.random() > 0.5:
            patch = np.flip(patch, axis=0)
        k = random.randint(0, 3)
        patch = np.rot90(patch, k, axes=(1, 2))
        return patch.copy()

    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        if self.train:
            patch = self.augment_patch(patch)
        
        patch = np.clip(patch, 0.0, 1.0)#->[0, 1]
        patch = np.expand_dims(patch, axis=0).astype(np.float32)
        return torch.tensor(patch), torch.tensor(label, dtype=torch.long)
    

class NoduleClassifier4(nn.Module):
    def __init__(self):
        super(NoduleClassifier4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),nn.BatchNorm3d(32),nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=2),nn.BatchNorm3d(64),nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),  
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.4), 
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )
        self.residual = nn.Sequential(
            nn.Conv3d(1, 256, kernel_size=1),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.features(x)
        x += residual
        x = self.classifier(x)  # طبقه‌بندی نهایی
        return x
    



class NoduleClassifier(nn.Module):
    def __init__(self):
        super(NoduleClassifier, self).__init__()
        self.features = nn.Sequential(nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),nn.MaxPool3d(kernel_size=2),ResidualBlock3D(32, 64, kernel_size=3, stride=1, padding=1),  # Residual Block
            nn.MaxPool3d(kernel_size=2),ResidualBlock3D(64, 128, kernel_size=3, stride=1, padding=1),  # Residual Block
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),  # Additional deeper layer
            nn.BatchNorm3d(512),nn.ReLU(inplace=True),nn.AdaptiveAvgPool3d(1))
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(512, 256),nn.ReLU(),nn.BatchNorm1d(256),nn.Dropout(p=0.5),nn.Linear(256, 128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(p=0.4),nn.Linear(128, 2))
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.residual_bn = nn.BatchNorm3d(out_channels)
    def forward(self, x):
        residual = self.residual(x)
        residual = self.residual_bn(residual)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out