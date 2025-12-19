import torch
import torch.nn as nn
import numpy as np

class DnCNN(nn.Module):
    """
    DnCNN: Deep Convolutional Neural Network for Image Denoising
    IMPROVED VERSION with better stability
    """
    def __init__(self, channels=3, num_layers=20, features=64):
        super(DnCNN, self).__init__()

        layers = []


        layers.append(nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))


        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))


        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=True))

        self.dncnn = nn.Sequential(*layers)


        self._initialize_weights()

    def forward(self, x):

        noise = self.dncnn(x)

        return torch.clamp(x - noise, 0, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def load_dnccnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Re‑create architecture exactly as during training
    model = DnCNN(channels=3, num_layers=15, features=48).to(device)

    # Load weights
    state_dict = torch.load("/Users/yousefshafik/Desktop/coen691/COEN691-project/Demo/server/models/dncnn/best_dncnn_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, device