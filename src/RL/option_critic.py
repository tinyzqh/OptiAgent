import OptiVerse
import numpy as np
from time import sleep
import matplotlib.pyplot as plt


# logsumexp() and expit() are used because they are
# numerically stable
# expit() is the sigmoid function
from scipy.special import logsumexp
from scipy.special import expit
import gymnasium as gym


class EpsGreedyPolicy:
    def __init__(self, rng, nstates, noptions, epsilon):
        self.rng = rng
        self.nstates = nstates
        self.noptions = noptions
        self.epsilon = epsilon
        self.Q_Omega_table = np.zeros((nstates, noptions))

    def Q_Omega(self, state, option=None):
        if option is None:
            return self.Q_Omega_table[state, :]
        else:
            return self.Q_Omega_table[state, option]

    def sample(self, state):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.noptions))
        else:
            return int(np.argmax(self.Q_Omega(state)))


class SoftmaxPolicy:
    def __init__(self, rng, lr, nstates, nactions, temperature=1.0):
        self.rng = rng
        self.lr = lr
        self.nstates = nstates
        self.nactions = nactions
        self.temperature = temperature
        self.weights = np.zeros((nstates, nactions))

    def Q_U(self, state, action=None):
        if action is None:
            return self.weights[state, :]
        else:
            return self.weights[state, action]

    def pmf(self, state):
        exponent = self.Q_U(state) / self.temperature
        return np.exp(exponent - logsumexp(exponent))

    def sample(self, state):
        return int(self.rng.choice(self.nactions, p=self.pmf(state)))

    def gradient(self):
        pass

    def update(self, state, action, Q_U):
        actions_pmf = self.pmf(state)
        self.weights[state, :] -= self.lr * actions_pmf * Q_U
        self.weights[state, action] += self.lr * Q_U


class SigmoidTermination:
    def __init__(self, rng, lr, nstates):
        self.rng = rng
        self.lr = lr
        self.nstates = nstates
        self.weights = np.zeros((nstates,))

    def pmf(self, state):
        return expit(self.weights[state])

    def sample(self, state):
        return int(self.rng.uniform() < self.pmf(state))

    def gradient(self, state):
        return self.pmf(state) * (1.0 - self.pmf(state)), state

    def update(self, state, advantage):
        magnitude, direction = self.gradient(state)
        self.weights[direction] -= self.lr * magnitude * advantage


class Critic:
    def __init__(self, lr, discount, Q_Omega_table, nstates, noptions, nactions):
        self.lr = lr
        self.discount = discount
        self.Q_Omega_table = Q_Omega_table
        self.Q_U_table = np.zeros((nstates, noptions, nactions))

    def cache(self, state, option, action):
        self.last_state = state
        self.last_option = option
        self.last_action = action
        self.last_Q_Omega = self.Q_Omega(state, option)

    def Q_Omega(self, state, option=None):
        if option is None:
            return self.Q_Omega_table[state, :]
        else:
            return self.Q_Omega_table[state, option]

    def Q_U(self, state, option, action):
        return self.Q_U_table[state, option, action]

    def A_Omega(self, state, option=None):
        advantage = self.Q_Omega(state) - np.max(self.Q_Omega(state))

        if option is None:
            return advantage
        else:
            return advantage[option]

    def update_Qs(self, state, option, action, reward, done, terminations):
        # One step target for Q_Omega
        target = reward
        if not done:
            beta_omega = terminations[self.last_option].pmf(state)
            target += self.discount * ((1.0 - beta_omega) * self.Q_Omega(state, self.last_option) + beta_omega * np.max(self.Q_Omega(state)))

        # Difference update
        tderror_Q_Omega = target - self.last_Q_Omega
        self.Q_Omega_table[self.last_state, self.last_option] += self.lr * tderror_Q_Omega

        tderror_Q_U = target - self.Q_U(self.last_state, self.last_option, self.last_action)
        self.Q_U_table[self.last_state, self.last_option, self.last_action] += self.lr * tderror_Q_U

        # Cache
        self.last_state = state
        self.last_option = option
        self.last_action = action
        if not done:
            self.last_Q_Omega = self.Q_Omega(state, option)


# env = gym.make("FourRoom-v0", seed=42)

# env = FourRooms()
# env.reset()

# Discount
discount = 0.99

# Learning rates - termination, intra-option, critic
lr_term = 0.25
lr_intra = 0.25
lr_critic = 0.5

# Epsilon for epsilon-greedy for policy over options
epsilon = 1e-1

# Temperature for softmax
temperature = 1e-2

# Number of runs
nruns = 10

# Number of episodes per run
nepisodes = 1000

# Maximum number of steps per episode
nsteps = 1000

# Number of options
noptions = 4

# Random number generator for reproducability
rng = np.random.RandomState(1234)

# The possible next goals (all in the lower right room)
possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]

# History of steps and average durations
history = np.zeros((nruns, nepisodes, 2))

option_terminations_list = []

for run in range(nruns):

    env = gym.make("FourRoom-v0", seed=42)

    nstates = env.observation_space.n
    nactions = env.action_space.n

    # Following three belong to the Actor

    # 1. The intra-option policies - linear softmax functions
    option_policies = [SoftmaxPolicy(rng, lr_intra, nstates, nactions, temperature) for _ in range(noptions)]

    # 2. The termination function - linear sigmoid function
    option_terminations = [SigmoidTermination(rng, lr_term, nstates) for _ in range(noptions)]

    # 3. The epsilon-greedy policy over options
    policy_over_options = EpsGreedyPolicy(rng, nstates, noptions, epsilon)

    # Critic
    critic = Critic(lr_critic, discount, policy_over_options.Q_Omega_table, nstates, noptions, nactions)

    print("Goal: ", env.unwrapped.goal)

    for episode in range(nepisodes):

        # Change goal location after 1000 episodes
        # Comment it for not doing transfer experiments
        if episode == 1000:
            env.goal = rng.choice(possible_next_goals)
            print("New goal: ", env.goal)

        state, info = env.reset()

        option = policy_over_options.sample(state)
        action = option_policies[option].sample(state)

        critic.cache(state, option, action)

        duration = 1
        option_switches = 0
        avg_duration = 0.0

        for step in range(nsteps):

            state, reward, terminations, truncations, _ = env.step(action)
            done = np.array((np.logical_or(terminations, truncations)))

            # Termination might occur upon entering new state
            if option_terminations[option].sample(state):
                option = policy_over_options.sample(state)
                option_switches += 1
                avg_duration += (1.0 / option_switches) * (duration - avg_duration)
                duration = 1

            action = option_policies[option].sample(state)

            # Critic update
            critic.update_Qs(state, option, action, reward, done, option_terminations)

            # Intra-option policy update with baseline
            Q_U = critic.Q_U(state, option, action)
            Q_U = Q_U - critic.Q_Omega(state, option)
            option_policies[option].update(state, action, Q_U)

            # Termination condition update
            option_terminations[option].update(state, critic.A_Omega(state, option))

            duration += 1

            if done:
                break

        history[run, episode, 0] = step
        history[run, episode, 1] = avg_duration

    option_terminations_list.append(option_terminations)

    # Plot stuff
    plt.figure(figsize=(20, 6))
    plt.subplot(121)
    plt.title("run: %s" % run)
    plt.xlabel("episodes")
    plt.ylabel("steps")
    plt.plot(np.mean(history[: run + 1, :, 0], axis=0))
    plt.grid(True)
    plt.subplot(122)
    plt.title("run: %s" % run)
    plt.xlabel("episodes")
    plt.ylabel("avg. option duration")
    plt.plot(np.mean(history[: run + 1, :, 1], axis=0))
    plt.grid(True)
    plt.savefig("fourrooms{}.png".format(episode))
