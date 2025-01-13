import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator

# Hyperparameters
latent_dim = 100
image_size = 64
channels = 1  # Grayscale images
batch_size = 128
epochs = 50
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())  # Should return True


# Data preparation for ECG dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize for single channel
])

# Replace with the path to your ECG dataset
# Replace with the actual path to your ECG dataset
dataset = ImageFolder(root="C:\\Users\\rishi\\OneDrive\\Desktop\\mini project\\ECG_Image_data", transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
discriminator = Discriminator(image_size=image_size, channels=channels).to(device)
generator = Generator(latent_dim=latent_dim, channels=channels).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))

def train():
    for epoch in range(epochs):
        for batch_idx, (real_imgs, _) in enumerate(data_loader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real images
            real_loss = criterion(discriminator(real_imgs), real_labels)

            # Fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_imgs = generator(noise)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)

            # Total loss and backpropagation
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            # Generator loss
            g_loss = criterion(discriminator(fake_imgs), real_labels)
            g_loss.backward()
            optimizer_G.step()

            # Print training progress
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(data_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # Save sample images
        save_image(fake_imgs.data[:25], f"images/{epoch + 1}.png", nrow=5, normalize=True)

if __name__ == "__main__":
    train()







