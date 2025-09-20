import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image


# ===============================
# Configuration & Hyperparameters
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets"

USE_CUDA = False
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE = 100
INPUT_DIM = 784  # 28x28 flattened MNIST image
HIDDEN_DIM = 400
LATENT_DIM = 200
LEARNING_RATE = 1e-3
EPOCHS = 30

# ===============================
# Dataset & DataLoader
# ===============================
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(DATASET_PATH, train=True, transform=transform, download=True)
test_dataset = MNIST(DATASET_PATH, train=False, transform=transform, download=True)

loader_kwargs = {"num_workers": 1, "pin_memory": True}
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)


# ===============================
# Model Components
# ===============================
class Encoder(nn.Module):
    """
    Gaussian MLP Encoder:
    Maps input images to a latent distribution (mean and log-variance).
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var


class Decoder(nn.Module):
    """
    Gaussian MLP Decoder:
    Reconstructs images from latent variables.
    """

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, z):
        h = self.activation(self.fc1(z))
        h = self.activation(self.fc2(h))
        x_hat = torch.sigmoid(self.fc_output(h))
        return x_hat


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) with reparameterization trick.
    """

    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick:
        z = mean + std * epsilon, with epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std).to(DEVICE)
        return mean + std * epsilon

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


# ===============================
# Loss Function
# ===============================
def vae_loss(x, x_hat, mean, log_var):
    """
    VAE loss = Reconstruction Loss + KL Divergence
    """
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kld


# ===============================
# Training
# ===============================
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
model = VariationalAutoencoder(encoder, decoder).to(DEVICE)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

print("Start training VAE...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.view(BATCH_SIZE, INPUT_DIM).to(DEVICE)

        optimizer.zero_grad()
        x_hat, mean, log_var = model(images)
        loss = vae_loss(images, x_hat, mean, log_var)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / (len(train_loader.dataset))
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}")

print("Training Finished!")


# ===============================
# Evaluation & Visualization
# ===============================
model.eval()


def save_single_image(x, idx, filename):
    """
    Save a single image from a batch as PNG.
    """
    x = x.view(BATCH_SIZE, 28, 28)
    plt.imshow(x[idx].cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


# Reconstruction
with torch.no_grad():
    for batch_idx, (images, _) in enumerate(test_loader):
        images = images.view(BATCH_SIZE, INPUT_DIM).to(DEVICE)
        recon_images, _, _ = model(images)
        break

save_single_image(images, idx=0, filename="original.png")
save_single_image(recon_images, idx=0, filename="reconstructed.png")

# Sampling from latent space
with torch.no_grad():
    noise = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
    generated_images = decoder(noise)

save_image(generated_images.view(BATCH_SIZE, 1, 28, 28), "generated_samples.png")
save_single_image(generated_images, idx=0, filename="sample_0.png")
save_single_image(generated_images, idx=12, filename="sample_12.png")
