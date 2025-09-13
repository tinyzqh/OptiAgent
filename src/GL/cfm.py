import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from torch.optim import AdamW
from pathlib import Path
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# https://github.com/Jackson-Kang/Pytorch-Conditional-Flow-Matching-Tutorial/blob/main/01_conditional-flow-matching.ipynb


# ===============================
# Configuration & Hyperparameters
# ===============================
DATASET = "MNIST"   # or "CIFAR10"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets"

GPU_ID = 2
USE_CUDA = False
DEVICE = torch.device(f"cuda:0" if USE_CUDA else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"

HIDDEN_DIM = 256
N_LAYERS = 8
HIDDEN_DIMS = [HIDDEN_DIM for _ in range(N_LAYERS)]

LR = 5e-5
SIGMA_MIN = 0.0
N_EPOCHS = 200
TRAIN_BATCH_SIZE = 128
INFER_BATCH_SIZE = 64
SEED = 1234

IMG_SIZE = (32, 32, 3) if DATASET == "CIFAR10" else (28, 28, 1)  # (H, W, C)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===============================
# Dataset & DataLoader
# ===============================
transform = transforms.Compose([transforms.ToTensor()])
kwargs = {"num_workers": 1, "pin_memory": True}

if DATASET == "CIFAR10":
    train_dataset = CIFAR10(DATASET_PATH, train=True, transform=transform, download=True)
    test_dataset = CIFAR10(DATASET_PATH, train=False, transform=transform, download=True)
else:
    train_dataset = MNIST(DATASET_PATH, train=True, transform=transform, download=True)
    test_dataset = MNIST(DATASET_PATH, train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=INFER_BATCH_SIZE, shuffle=False, **kwargs)


# ===============================
# Model Components
# ===============================
class ConvBlock(nn.Conv2d):
    """
    Conv2D Block with optional GroupNorm and activation.
    Args:
        x: (N, C_in, H, W)
    Returns:
        y: (N, C_out, H, W)
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


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching (CFM) Model.
    Learns a vector field that transports Gaussian noise to target data distribution.
    """
    def __init__(self, image_resolution, hidden_dims, sigma_min=0.):
        super().__init__()
        _, _, img_C = image_resolution

        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7)
        self.time_project = nn.Sequential(
            ConvBlock(1, hidden_dims[0], kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[1], kernel_size=1)
        )

        self.convs = nn.ModuleList()
        for idx in range(len(hidden_dims)):
            dilation = 3 ** ((idx - 1) // 2) if idx > 0 else 1
            in_channels = hidden_dims[idx - 1] if idx > 0 else hidden_dims[0]
            self.convs.append(
                ConvBlock(in_channels, hidden_dims[idx], kernel_size=3,
                          dilation=dilation, activation_fn=True, gn=True, gn_groups=8)
            )

        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_C, kernel_size=3)
        self.sigma_min = sigma_min

    def get_velocity(self, x_0, x_1):
        """
        Target velocity (vector field).
        """
        return x_1 - (1 - self.sigma_min) * x_0

    def interpolate(self, x_0, x_1, t):
        """
        Linear interpolation between Gaussian noise (x_0) and target sample (x_1).
        """
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1

    def forward(self, x, t):
        """
        Estimate vector field v_t given x_t and time embedding.
        """
        time_embedding = self.time_project(t)
        y = self.in_project(x)
        for conv in self.convs:
            y = conv(y, time_embedding, residual=True)
        return self.out_project(y)

    @torch.no_grad()
    def sample(self, t_steps, shape, device):
        """
        Transport noise x_0 ~ N(0, I) to x_1 ~ p_data using learned vector field.
        """
        B, C, W, H = shape
        x_0 = torch.randn(size=shape, device=device)
        t_vals = torch.linspace(0, 1, t_steps, device=device)
        delta = 1.0 / (t_steps - 1)

        x_hat = x_0
        for i in range(t_steps - 1):
            t_cur = t_vals[i].view(-1, 1, 1, 1)
            velocity_pred = self(x_hat, t_cur)
            x_hat = x_hat + velocity_pred * delta
        return x_hat


# ===============================
# Training Setup
# ===============================
model = ConditionalFlowMatching(image_resolution=IMG_SIZE,
                                hidden_dims=HIDDEN_DIMS,
                                sigma_min=SIGMA_MIN).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99))


# ===============================
# Training Loop
# ===============================
print("Start training CFM...")
model.train()

for epoch in range(N_EPOCHS):
    total_loss = 0
    for batch_idx, (x_1, _) in enumerate(train_loader):
        optimizer.zero_grad()

        # Sample Gaussian noise and target data
        x_1 = x_1.to(DEVICE)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], 1, 1, 1, device=DEVICE)

        # Interpolation and velocity targets
        x_t = model.interpolate(x_0, x_1, t)
        velocity_target = model.get_velocity(x_0, x_1)
        velocity_pred = model(x_t, t)

        # CFM loss
        loss = ((velocity_pred - velocity_target) ** 2).mean()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            print(f"\tBatch {batch_idx}: Loss = {loss.item():.6f}, GradNorm = {grad_norm:.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{N_EPOCHS}] - Avg CFM Loss: {avg_loss:.6f}")

print("Training Finished!")


# ===============================
# Visualization Utilities
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
    Save a grid of images as PNG.
    """
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


# ===============================
# Inference & Sampling
# ===============================
model.eval()
B, (W, H, C) = 1, IMG_SIZE

for steps in [2, 5, 10, 25, 50]:
    x_samples = model.sample(t_steps=steps, shape=[B, C, W, H], device=DEVICE)
    save_single_image(x_samples, idx=0, filename=f"sample_steps_{steps}.png")

# Full batch generation
B = INFER_BATCH_SIZE
steps = 50
generated_images = model.sample(t_steps=steps, shape=[B, C, W, H], device=DEVICE)

save_grid(generated_images, filename="generated_grid.png", title="Generated Images")
save_grid(x_1[:INFER_BATCH_SIZE], filename="ground_truth.png", title="Ground-truth Images")
