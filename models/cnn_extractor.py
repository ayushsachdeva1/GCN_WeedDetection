import torch.nn as nn
import torchvision

class CNN_Model(nn.Module):
    def __init__(self, feature_dim = 256):
        super().__init__()
        self.resnet = torchvision.models.resnet101(pretrained=True)
        self.fc = nn.Linear(1000, feature_dim)

    def forward(self, x):
        feat = self.resnet(x)
        return self.fc(feat)