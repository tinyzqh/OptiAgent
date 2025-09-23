import os
import torch
import OptiVerse
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Distribution, Normal


class GaussianPolicy(nn.Module):
    """
    Gaussian Policy
    """

    def __init__(self, state_dim, action_dim, max_action, device, hidden_sizes=[256, 256], layer_norm=False):
        super(GaussianPolicy, self).__init__()

        self.layer_norm = layer_norm
        self.base_fc = []
        last_size = state_dim
        for next_size in hidden_sizes:
            self.base_fc += [nn.Linear(last_size, next_size), nn.LayerNorm(next_size) if layer_norm else nn.Identity(), nn.ReLU(inplace=True)]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)

        last_hidden_size = hidden_sizes[-1]
        self.last_fc_mean = nn.Linear(last_hidden_size, action_dim)
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)

        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20

        self.device = device

    def forward(self, state):

        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(self.LOG_SIG_MIN, self.LOG_SIG_MAX).exp()

        a_normal = Normal(mean, std, self.device)
        action = a_normal.rsample()
        log_prob = a_normal.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def log_prob(self, state, action):
        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(self.LOG_SIG_MIN, self.LOG_SIG_MAX).exp()

        a_normal = Normal(mean, std, self.device)
        log_prob = a_normal.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return log_prob

    def sample(self, state, reparameterize=False, deterministic=False):

        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(self.LOG_SIG_MIN, self.LOG_SIG_MAX).exp()

        if deterministic:
            action = mean
        else:
            a_normal = Normal(mean, std, self.device)
            if reparameterize:
                action = a_normal.rsample()
            else:
                action = a_normal.sample()

        return action


def test_bandit_bc_mle():
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

    actor = GaussianPolicy(state_dim=2, action_dim=2, max_action=1.0, device="cpu", hidden_sizes=[128, 128])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    for it in range(100 * 100):
        # Sample a batch of data from the environment
        state, action_samples, reward, _ = env.step(100)

        # Actor Training
        log_pi = actor.log_prob(torch.tensor(state), torch.tensor(action_samples))

        actor_loss = -log_pi.mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
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
    ax[1].set_xlim(-2.5, 2.5)
    ax[1].set_ylim(-2.5, 2.5)
    ax[1].set_xlabel("x", fontsize=20)
    ax[1].set_ylabel("y", fontsize=20)
    ax[1].set_title("BC-MLE", fontsize=25)

    # Improve layout and save figure
    fig.tight_layout()
    save_path = os.path.join(img_dir, f"ILBanditTask_BC_MLP_Seed_{seed}.pdf")
    fig.savefig(save_path)
    print(f"Saved action samples plot to {save_path}")

    # Close environment
    env.close()


if __name__ == "__main__":
    test_bandit_bc_mle()
