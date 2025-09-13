import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
import math
from pathlib import Path
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# https://github.com/Jackson-Kang/Pytorch-Diffusion-Model-Tutorial/blob/main/01_denoising_diffusion_probabilistic_model.ipynb


# ===============================
# Configuration & Hyperparameters
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets"

USE_CUDA = False
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

DATASET = "MNIST"   # Change to "CIFAR10" if needed
IMG_SIZE = (32, 32, 3) if DATASET == "CIFAR10" else (28, 28, 1)  # (H, W, C)

TIME_EMB_DIM = 256
N_LAYERS = 8
HIDDEN_DIM = 256
N_TIMESTEPS = 1000
BETA_RANGE = [1e-4, 2e-2]

TRAIN_BATCH_SIZE = 128
INFER_BATCH_SIZE = 64
LR = 5e-5
EPOCHS = 200

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

hidden_dims = [HIDDEN_DIM for _ in range(N_LAYERS)]


# ===============================
# Dataset & DataLoader
# ===============================
transform = transforms.Compose([transforms.ToTensor()])
kwargs = {"num_workers": 1, "pin_memory": True}

if DATASET == "CIFAR10":
    train_dataset = CIFAR10(DATASET_PATH, transform=transform, train=True, download=True)
    test_dataset = CIFAR10(DATASET_PATH, transform=transform, train=False, download=True)
else:
    train_dataset = MNIST(DATASET_PATH, transform=transform, train=True, download=True)
    test_dataset = MNIST(DATASET_PATH, transform=transform, train=False, download=True)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=INFER_BATCH_SIZE, shuffle=False, **kwargs)


# ===============================
# Model Components
# ===============================
class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding for diffusion timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConvBlock(nn.Conv2d):
    """
    Conv2D Block with optional GroupNorm and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=False, drop_rate=0.,
                 stride=1, padding="same", dilation=1, groups=1, bias=True, gn=False, gn_groups=8):
        
        if padding == "same":
            padding = kernel_size // 2 * dilation

        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias)

        self.activation_fn = nn.SiLU() if activation_fn else None
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None

    def forward(self, x, time_embedding=None, residual=False):
        if residual:
            x = x + time_embedding
            y = x + super().forward(x)
        else:
            y = super().forward(x)

        if self.group_norm is not None:
            y = self.group_norm(y)
        if self.activation_fn is not None:
            y = self.activation_fn(y)

        return y


class Denoiser(nn.Module):
    """
    U-Net-like denoiser for predicting noise ε at timestep t.
    """
    def __init__(self, image_resolution, hidden_dims, diffusion_time_embedding_dim=256, n_times=1000):
        super().__init__()
        _, _, img_C = image_resolution

        self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim)
        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7)

        self.time_project = nn.Sequential(
            ConvBlock(diffusion_time_embedding_dim, hidden_dims[0], kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[0], kernel_size=1)
        )

        self.convs = nn.ModuleList()
        for idx in range(len(hidden_dims)):
            dilation = 3 ** ((idx - 1) // 2) if idx > 0 else 1
            self.convs.append(
                ConvBlock(hidden_dims[idx - 1] if idx > 0 else hidden_dims[0],
                          hidden_dims[idx],
                          kernel_size=3,
                          dilation=dilation,
                          activation_fn=True,
                          gn=True,
                          gn_groups=8)
            )

        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_C, kernel_size=3)

    def forward(self, perturbed_x, timestep):
        diffusion_embedding = self.time_embedding(timestep)
        diffusion_embedding = self.time_project(diffusion_embedding.unsqueeze(-1).unsqueeze(-2))

        y = self.in_project(perturbed_x)
        for conv in self.convs:
            y = conv(y, diffusion_embedding, residual=True)

        return self.out_project(y)


# ===============================
# Diffusion Process
# ===============================
class Diffusion(nn.Module):
    """
    Diffusion wrapper for forward noising and reverse denoising.
    """
    def __init__(self, model, image_resolution, n_times=1000, beta_minmax=[1e-4, 2e-2], device="cuda"):
        super().__init__()
        self.n_times = n_times
        self.img_H, self.img_W, self.img_C = image_resolution
        self.model = model
        self.device = device

        beta_1, beta_T = beta_minmax
        betas = torch.linspace(beta_1, beta_T, n_times).to(device)
        self.sqrt_betas = torch.sqrt(betas)

        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def scale_to_minus_one_to_one(self, x):
        return x * 2 - 1

    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

    def make_noisy(self, x_zeros, t):
        epsilon = torch.randn_like(x_zeros).to(self.device)
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
        return noisy_sample.detach(), epsilon

    def forward(self, x_zeros):
        x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        B, _, _, _ = x_zeros.shape
        t = torch.randint(0, self.n_times, (B,)).long().to(self.device)
        perturbed_images, epsilon = self.make_noisy(x_zeros, t)
        pred_epsilon = self.model(perturbed_images, t)
        return perturbed_images, epsilon, pred_epsilon

    def denoise_at_t(self, x_t, timestep, t):
        B, _, _, _ = x_t.shape
        z = torch.randn_like(x_t).to(self.device) if t > 1 else torch.zeros_like(x_t).to(self.device)

        epsilon_pred = self.model(x_t, timestep)
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)

        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * epsilon_pred) + sqrt_beta * z
        return x_t_minus_1.clamp(-1., 1)

    def sample(self, N):
        x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device)
        for t in range(self.n_times - 1, -1, -1):
            timestep = torch.tensor([t]).repeat(N).long().to(self.device)
            x_t = self.denoise_at_t(x_t, timestep, t)
        return self.reverse_scale_to_zero_to_one(x_t)


# ===============================
# Training Setup
# ===============================
model = Denoiser(image_resolution=IMG_SIZE,
                 hidden_dims=hidden_dims,
                 diffusion_time_embedding_dim=TIME_EMB_DIM,
                 n_times=N_TIMESTEPS).to(DEVICE)

diffusion = Diffusion(model, image_resolution=IMG_SIZE,
                      n_times=N_TIMESTEPS,
                      beta_minmax=BETA_RANGE,
                      device=DEVICE).to(DEVICE)

optimizer = Adam(diffusion.parameters(), lr=LR)
criterion = nn.MSELoss()


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


print("Number of model parameters:", count_parameters(diffusion))


# ===============================
# Utility Functions
# ===============================
def save_single_image(x, idx, filename):
    """
    Save a single image from a batch as PNG.
    """
    img = x[idx].detach().cpu()
    if img.shape[0] == 1:  # grayscale
        plt.imshow(img.squeeze(), cmap="gray")
    else:
        plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_grid(x, filename, title="Generated Samples"):
    """
    Save a grid of images.
    """
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


# ===============================
# Training Loop
# ===============================
print("Start training DDPM...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        x = x.to(DEVICE)
        noisy_input, epsilon, pred_epsilon = diffusion(x)
        loss = criterion(pred_epsilon, epsilon)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Denoising Loss: {avg_loss:.6f}")

print("Training Finished!")


# ===============================
# Inference & Visualization
# ===============================
model.eval()
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.to(DEVICE)
        perturbed_images, _, _ = diffusion(x)
        perturbed_images = diffusion.reverse_scale_to_zero_to_one(perturbed_images)
        break

with torch.no_grad():
    generated_images = diffusion.sample(N=INFER_BATCH_SIZE)

# Save individual images
save_single_image(perturbed_images, idx=0, filename="perturbed_0.png")
save_single_image(perturbed_images, idx=1, filename="perturbed_1.png")
save_single_image(generated_images, idx=0, filename="generated_0.png")
save_single_image(generated_images, idx=1, filename="generated_1.png")

# Save grids
save_grid(perturbed_images, filename="perturbed_grid.png", title="Perturbed Images")
save_grid(generated_images, filename="generated_grid.png", title="Generated Images")
save_grid(x[:INFER_BATCH_SIZE], filename="ground_truth.png", title="Ground-truth Images")
