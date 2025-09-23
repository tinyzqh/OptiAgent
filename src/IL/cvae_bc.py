import os
import torch
import OptiVerse
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Distribution, Normal


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device, hidden_dim=256):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def sample(self, state):
        return self.decode(state)


def test_bandit_bc_vae():
    """
    Run a simple test of the ILBanditTask environment:
    - Initialize the environment
    - Sample a batch of points
    - Visualize the sampled actions distribution
    - Save the plot as a PDF
    """
    seed = 2025
    num_points = 10000
    batch_size = 1000

    # Create environment
    env = gym.make("ILBanditTask-v0", num_points=num_points, seed=seed)

    # Reset environment (important for reproducibility)
    env.reset(seed=seed)

    vae = VAE(state_dim=2, action_dim=2, latent_dim=4, max_action=1.0, device="cpu", hidden_dim=128)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=3e-4)

    for it in range(100 * 100):
        # Sample a batch of data from the environment
        state, action_samples, reward, _ = env.step(100)

        # Variational Auto-Encoder Training
        recon, mean, std = vae(torch.tensor(state), torch.tensor(action_samples))
        recon_loss = F.mse_loss(recon, torch.tensor(action_samples))
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()

    # Visualization settings
    axis_lim = 1.1
    img_dir = "images/IL/ILBanditTask"
    os.makedirs(img_dir, exist_ok=True)  # Ensure directory exists

    fig, ax = plt.subplots(1, 2, figsize=(5.5 * 2, 5.5))

    # Plot sampled actions
    ax[0].scatter(action_samples[:, 0], action_samples[:, 1], alpha=0.3, s=10, c="blue")

    # Set plot limits and labels
    ax[0].set_xlim(-axis_lim, axis_lim)
    ax[0].set_ylim(-axis_lim, axis_lim)
    ax[0].set_xlabel("x", fontsize=14)
    ax[0].set_ylabel("y", fontsize=14)
    ax[0].set_title("Ground Truth Action Samples", fontsize=16)

    new_state = torch.zeros((1000, 2), device="cpu")
    new_action = vae.sample(new_state)
    new_action = new_action.detach().cpu().numpy()
    ax[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
    ax[1].set_xlim(-axis_lim, axis_lim)
    ax[1].set_ylim(-axis_lim, axis_lim)
    ax[1].set_xlabel("x", fontsize=20)
    ax[1].set_ylabel("y", fontsize=20)
    ax[1].set_title("BC-VAE", fontsize=25)

    # Improve layout and save figure
    fig.tight_layout()
    save_path = os.path.join(img_dir, f"ILBanditTask_BC_VAE_Seed_{seed}.pdf")
    fig.savefig(save_path)
    print(f"Saved action samples plot to {save_path}")

    # Close environment
    env.close()


if __name__ == "__main__":
    test_bandit_bc_vae()
