import os
import torch
import OptiVerse
import numpy as np
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt

import torch.nn.functional as F

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.IL.diffusion import Diffusion

# from toy_experiments.diffusion import Diffusion
from src.IL.helpers import EMA
from src.IL.helpers import SinusoidalPosEmb


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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class QL_Diffusion(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount,
        tau,
        max_q_backup=False,
        eta=1.0,
        model_type="MLP",
        beta_schedule="linear",
        n_timesteps=100,
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=3e-4,
        hidden_dim=32,
        r_fun=None,
        mode="whole_grad",
    ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device, hidden_dim=hidden_dim)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action, beta_schedule=beta_schedule, n_timesteps=n_timesteps).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        if r_fun is None:
            self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

        self.r_fun = r_fun
        self.mode = mode

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, replay_buffer, iterations, batch_size=100):

        for step in range(iterations):
            # Sample replay buffer / batch
            state, action, reward = replay_buffer.sample(batch_size)

            if self.r_fun is None:
                current_q1, current_q2 = self.critic(state, action)
                critic_loss = F.mse_loss(current_q1, reward) + F.mse_loss(current_q2, reward)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)

            if self.mode == "whole_grad":
                new_action = self.actor(state)
            elif self.mode == "t_middle":
                new_action = self.actor.sample_t_middle(state)
            elif self.mode == "t_last":
                new_action = self.actor.sample_t_last(state)
            elif self.mode == "last_few":
                new_action = self.actor.sample_last_few(state)

            if self.r_fun is None:
                q1_new_action, q2_new_action = self.critic(state, new_action)
                if np.random.uniform() > 0.5:
                    lmbda = self.eta / q2_new_action.abs().mean().detach()
                    q_loss = -lmbda * q1_new_action.mean()
                else:
                    lmbda = self.eta / q1_new_action.abs().mean().detach()
                    q_loss = -lmbda * q2_new_action.mean()
            else:
                q_new_action = self.r_fun(new_action)
                lmbda = self.eta / q_new_action.abs().mean().detach()
                q_loss = -lmbda * q_new_action.mean()

            actor_loss = bc_loss + q_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor.step_frozen()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            self.step += 1

        # Logging
        return bc_loss.item(), q_loss.item()

    def sample_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        # with torch.no_grad():
        #     action = self.actor.sample(state_rpt)
        #     q_value = self.critic_target.q_min(state_rpt, action).flatten()
        #     idx = torch.multinomial(F.softmax(q_value), 1)
        # return action[idx].cpu().data.numpy().flatten()
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f"{dir}/actor.pth")
        torch.save(self.critic.state_dict(), f"{dir}/critic.pth")

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f"{dir}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{dir}/critic.pth"))


def test_bandit_ql_diffusion():
    """
    Run a simple test of the ILBanditTask environment:
    - Initialize the environment
    - Sample a batch of points
    - Visualize the sampled actions distribution
    - Save the plot as a PDF
    """
    eta = 2.5
    seed = 2025
    num_points = 10000
    batch_size = 1000

    # Create environment
    env = gym.make("ILBanditRewardTask-v0", num_points=num_points, seed=seed)

    # Reset environment (important for reproducibility)
    env.reset(seed=seed)

    actor = QL_Diffusion(
        state_dim=2,
        action_dim=2,
        max_action=1.0,
        device="cpu",
        discount=0.99,
        tau=0.005,
        eta=2.5,
        beta_schedule="vp",
        n_timesteps=50,
        model_type="MLP",
        hidden_dim=128,
        lr=0.0003,
        r_fun=None,
        mode="whole_grad",
    )

    # actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    # critic = Critic(state_dim=2, action_dim=2, hidden_dim=128).to("cpu")
    # critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    for epoch in range(1000):
        for it in range(100):
            # Sample a batch of data from the environment
            state, action_samples, reward, _ = env.step(100)

            current_q1, current_q2 = actor.critic(torch.tensor(state), torch.tensor(action_samples))
            critic_loss = F.mse_loss(current_q1, torch.tensor(reward)) + F.mse_loss(current_q2, torch.tensor(reward))

            actor.critic_optimizer.zero_grad()
            critic_loss.backward()
            actor.critic_optimizer.step()

            """ Policy Training """
            bc_loss = actor.actor.loss(torch.tensor(action_samples), torch.tensor(state))

            if actor.mode == "whole_grad":
                new_action = actor.actor(torch.tensor(state))
            elif actor.mode == "t_middle":
                new_action = actor.actor.sample_t_middle(torch.tensor(state))
            elif actor.mode == "t_last":
                new_action = actor.actor.sample_t_last(torch.tensor(state))
            elif actor.mode == "last_few":
                new_action = actor.actor.sample_last_few(torch.tensor(state))

            # log_pi = actor.log_prob(torch.tensor(state), torch.tensor(action_samples))
            # new_action, _ = actor(torch.tensor(state))
            q1_new_action, q2_new_action = actor.critic(torch.tensor(state), new_action)
            if np.random.uniform() > 0.5:
                lmbda = eta / q2_new_action.abs().mean().detach()
                q_loss = -lmbda * q1_new_action.mean()
            else:
                lmbda = eta / q1_new_action.abs().mean().detach()
                q_loss = -lmbda * q2_new_action.mean()

            actor_loss = bc_loss + q_loss
            actor.actor_optimizer.zero_grad()
            actor_loss.backward()
            # q_loss.backward()
            actor.actor_optimizer.step()
            actor.actor.step_frozen()

            if actor.step % actor.update_ema_every == 0:
                actor.step_ema()

            actor.step += 1

        if epoch % 100 == 0:
            print(f"QL-Diffusion Epoch: {epoch} B_loss {bc_loss.item()} Q_loss {q_loss.item()}")

    # Visualization settings
    axis_lim = 1.1
    img_dir = "images/QL/ILBanditTask"
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
    new_action = actor.actor.sample(new_state)
    new_action = new_action.detach().cpu().numpy()
    ax[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
    ax[1].set_xlim(-2.5, 2.5)
    ax[1].set_ylim(-2.5, 2.5)
    ax[1].set_xlabel("x", fontsize=20)
    ax[1].set_ylabel("y", fontsize=20)
    ax[1].set_title("BC-Diffusion", fontsize=25)

    # Improve layout and save figure
    fig.tight_layout()
    save_path = os.path.join(img_dir, f"ILBanditTask_QL_Diffusion_Seed_{seed}.pdf")
    fig.savefig(save_path)
    print(f"Saved action samples plot to {save_path}")

    # Close environment
    env.close()


if __name__ == "__main__":
    test_bandit_ql_diffusion()
