import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),  # (1x1) -> (4x4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (4x4) -> (8x8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (8x8) -> (16x16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (16x16) -> (32x32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),  # (32x32) -> (64x64)
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

if __name__ == "__main__":
    # Test the Generator
    noise = torch.randn(16, 100, 1, 1)  # Batch of 16 noise vectors (latent_dim=100)
    model = Generator()
    print(model(noise).shape)  # Output: torch.Size([16, 3, 64, 64])
