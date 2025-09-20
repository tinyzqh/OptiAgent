import os
import torch
import OptiVerse
import torch.nn as nn
import gymnasium as gym
from torch.optim import Adam
import matplotlib.pyplot as plt


class Encoder(nn.Module):
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
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, z):
        h = self.activation(self.fc1(z))
        h = self.activation(self.fc2(h))
        x_hat = self.fc_output(h)
        return x_hat


class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


def vae_loss(x, x_hat, mean, log_var, beta=1.0):
    """
    VAE loss = Reconstruction Loss + KL Divergence
    """
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kld


def kl_annealing(epoch, num_epochs, mode="linear"):
    """
    KL annealing schedule:
    - linear: beta 从 0 -> 1 线性增长
    - cos:    beta 用余弦曲线平滑增长
    """
    if mode == "linear":
        return min(1.0, epoch / num_epochs)
    elif mode == "cos":
        import math

        return (1 - math.cos(math.pi * epoch / num_epochs)) / 2
    else:
        return 1.0  # constant


if __name__ == "__main__":
    INPUT_DIM = 2
    HIDDEN_DIM = 4
    LATENT_DIM = 4
    LEARNING_RATE = 1e-3

    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
    model = VariationalAutoencoder(encoder, decoder)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train_len = 10
    EPOCHS = 30
    seed = 2025
    num_points = 10000
    env = gym.make("ILBanditTask-v0", num_points=num_points, seed=seed)
    env.reset(seed=seed)

    print("Start training VAE...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        beta = kl_annealing(epoch + 1, EPOCHS, mode="linear")  # KL 权重
        for i in range(train_len):
            state, action_samples, reward, _ = env.step(300)
            action_samples = torch.tensor(action_samples)
            mask = (action_samples > 0).all(dim=1)
            action_samples = action_samples[mask]

            optimizer.zero_grad()
            x_hat, mean, log_var = model(action_samples)
            loss = vae_loss(action_samples, x_hat, mean, log_var, beta)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / (train_len)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}")

    print("Training Finished!")

    model.eval()
    with torch.no_grad():
        state, action_samples, reward, _ = env.step(1000)
        action_samples = torch.tensor(action_samples)
        recon_x, mean, log_var = model(action_samples)

    # Visualization settings
    axis_lim = 1.1
    img_dir = "images/IL/ILBanditTask"
    os.makedirs(img_dir, exist_ok=True)  # Ensure directory exists

    # Plot sampled actions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(recon_x[:, 0], recon_x[:, 1], alpha=0.3, s=10, c="blue")

    # Set plot limits and labels
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_title("VAE Generate Action Samples", fontsize=16)

    # Improve layout and save figure
    fig.tight_layout()
    save_path = os.path.join(img_dir, f"ILBanditTask_Generate_Data_Seed_{seed}.pdf")
    fig.savefig(save_path)
    print(f"Saved action samples plot to {save_path}")
