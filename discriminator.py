import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_size=64, channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),  # (64x64) -> (32x32)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (32x32) -> (16x16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (16x16) -> (8x8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (8x8) -> (4x4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # (4x4) -> (1x1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)

if __name__ == "__main__":
    # Test the Discriminator
    img = torch.randn(16, 3, 64, 64)  # Batch of 16 images (3 channels, 64x64)
    model = Discriminator()
    print(model(img).shape)  # Output: torch.Size([16, 1])

