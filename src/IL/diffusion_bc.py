import os
import math
import torch
import OptiVerse
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Distribution, Normal

from src.IL.diffusion import Diffusion


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    """
    MLP Model
    """

    def __init__(self, state_dim, action_dim, device, t_dim=4, hidden_dim=32):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(SinusoidalPosEmb(t_dim), nn.Linear(t_dim, t_dim * 2), nn.Mish(), nn.Linear(t_dim * 2, t_dim))

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim), nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


def test_bandit_bc_diffusion():
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

    model = MLP(state_dim=2, action_dim=2, device="cpu", t_dim=4, hidden_dim=128)

    actor = Diffusion(state_dim=2, action_dim=2, model=model, max_action=1.0, beta_schedule="vp", n_timesteps=50)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    for it in range(100 * 1000):
        # Sample a batch of data from the environment
        state, action_samples, reward, _ = env.step(100)

        # Actor Training
        loss = actor.loss(torch.tensor(action_samples), torch.tensor(state))
        actor_optimizer.zero_grad()
        loss.backward()
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
    ax[1].set_title("BC-Diffusion", fontsize=25)

    # Improve layout and save figure
    fig.tight_layout()
    save_path = os.path.join(img_dir, f"ILBanditTask_BC_Diffusion_Seed_{seed}.pdf")
    fig.savefig(save_path)
    print(f"Saved action samples plot to {save_path}")

    # Close environment
    env.close()


if __name__ == "__main__":
    test_bandit_bc_diffusion()
