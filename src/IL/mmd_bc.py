import os
import torch
import OptiVerse
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

    def decode_multiple(self, state, z=None, num_decode=10):
        """Decode 10 samples atleast"""
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a)), self.d3(a)

    def sample(self, state):
        return self.decode(state)


class RegularActor(nn.Module):
    """A probabilistic actor which does regular stochastic mapping of actions from states"""

    def __init__(self, state_dim, action_dim, max_action, device, hidden_dim=256):
        super(RegularActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.device = device

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        z = mean_a + std_a * torch.randn_like(std_a)
        return self.max_action * torch.tanh(z)

    def sample_multiple(self, state, num_sample=10):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        # This trick stabilizes learning (clipping gaussian to a smaller range)
        z = mean_a.unsqueeze(1) + std_a.unsqueeze(1) * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))).to(self.device).clamp(-0.5, 0.5)
        return self.max_action * torch.tanh(z), z

    def log_pis(self, state, action=None, raw_action=None):
        """Get log pis for the model."""
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
        if raw_action is None:
            raw_action = atanh(action)
        else:
            action = torch.tanh(raw_action)
        log_normal = normal_dist.log_prob(raw_action)
        log_pis = log_normal.sum(-1)
        log_pis = log_pis - (1.0 - action ** 2).clamp(min=1e-6).log().sum(-1)
        return log_pis

    def sample(self, state):
        return self.forward(state)


def mmd_loss_laplacian(samples1, samples2, sigma=0.2):
    """MMD constraint with Laplacian kernel for support matching"""
    # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
    diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
    diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
    diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
    diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
    return overall_loss


def test_bandit_bc_mmd():
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

    actor = RegularActor(state_dim=2, action_dim=2, max_action=1.0, device="cpu", hidden_dim=128)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    for it in range(100 * 100):
        # Sample a batch of data from the environment
        state, action_samples, reward, _ = env.step(100)

        recon, mean, std = vae(torch.tensor(state), torch.tensor(action_samples))
        recon_loss = F.mse_loss(recon, torch.tensor(action_samples))
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()

        num_samples = 10
        sampled_actions, raw_sampled_actions = vae.decode_multiple(torch.tensor(state), num_decode=num_samples)  # B x N x d
        actor_actions, raw_actor_actions = actor.sample_multiple(torch.tensor(state), num_sample=num_samples)  # num)

        mmd_loss = mmd_loss_laplacian(raw_sampled_actions.detach(), raw_actor_actions, sigma=20).mean()
        actor_optimizer.zero_grad()
        mmd_loss.backward()
        actor_optimizer.step()

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
    new_action = actor.sample(new_state)
    new_action = new_action.detach().cpu().numpy()
    ax[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
    ax[1].set_xlim(-axis_lim, axis_lim)
    ax[1].set_ylim(-axis_lim, axis_lim)
    ax[1].set_xlabel("x", fontsize=20)
    ax[1].set_ylabel("y", fontsize=20)
    ax[1].set_title("BC-MMD", fontsize=25)

    # Improve layout and save figure
    fig.tight_layout()
    save_path = os.path.join(img_dir, f"ILBanditTask_BC_MMD_Seed_{seed}.pdf")
    fig.savefig(save_path)
    print(f"Saved action samples plot to {save_path}")

    # Close environment
    env.close()


if __name__ == "__main__":
    test_bandit_bc_mmd()
