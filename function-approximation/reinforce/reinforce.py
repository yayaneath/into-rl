import gym
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

RENDER_ENV = True

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, int(input_size * 10))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(input_size * 10), output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return F.softmax(out, dim=0)

def pick_action(policy, state):
    probs = policy(state) # Get actions probabilities
    cat_dist = Categorical(probs) # Create categorical distribution
    action = cat_dist.sample() # Sample an action

    # Return action and its log probability for loss
    return action.item(), cat_dist.log_prob(action)

def update_policy(policy, optimiser, rewards, log_probs, gamma):
    eps = 1e-8 # Avoid divison by zero
    step_return = 0.0 # Gt
    loss = []
    returns = []
    episode_length = len(rewards)

    # Calculate return from the end of the episode to the start
    for reward in reversed(rewards):
        step_return = reward + gamma * step_return
        returns.insert(0, step_return)

    returns = torch.tensor(returns)

    # Returns are normalised for stability of the learning
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for i in range(episode_length):
        loss.append(-log_probs[i] * returns[i])

    # Backward grads
    optimiser.zero_grad()
    loss = torch.stack(loss).sum()
    loss.backward()
    optimiser.step()

def reinforce(env, num_episodes, gamma, learning_rate):
    features_size = env.observation_space.shape[0]
    act_space_size = env.action_space.n

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy = Net(features_size, act_space_size).to(device)
    optimiser = optim.Adam(policy.parameters(), lr=learning_rate)
    rewards_evolution = []

    for ep in range(num_episodes + 1):
        finished = False
        ep_reward = 0.0
        rewards = []
        log_probs = []

        # Initialize s
        obs = env.reset()
        obs = torch.from_numpy(obs).to(device, dtype=torch.float)

        while not finished:
            if RENDER_ENV:
                env.render()

            # Pick an action
            action, action_log_prob = pick_action(policy, obs)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            # Save data for loss function
            log_probs.append(action_log_prob)
            rewards.append(reward)
            
            # Prepare next step
            obs = torch.from_numpy(new_obs).to(device, dtype=torch.float)
            ep_reward += reward

        rewards_evolution.append(ep_reward)

        # Now that the episode finished, calculate REINFORCE update to policy
        update_policy(policy, optimiser, rewards, log_probs, gamma)

        if (ep % 1 == 0):
            print('Episode', ep, '- Total reward:', ep_reward)

    return policy, rewards_evolution

if __name__ == '__main__':
    env = gym.make('CartPole-v0') #'MountainCar-v0' #'CartPole-v0'
    num_episodes = 2000
    gamma = 0.99
    learning_rate = 0.001

    start_time = time.time()
    policy, rewards = reinforce(env, num_episodes, gamma, learning_rate)
    end_time = time.time()

    print('REINFORCE took', end_time - start_time, 'seconds')

    file_name = 'reinforce-' + str(np.mean(rewards)) + '-' + str(end_time)
    torch.save(policy.state_dict(), file_name)

    _, ax = plt.subplots()

    window_size = 25
    stride = 1
    window_avg = [ np.mean(rewards[i : i + window_size]) for i in range(0, len(rewards), stride) if i + window_size <= len(rewards) ]

    ax.plot(window_avg)
    ax.set(xlabel='episode (avg 25)', ylabel='total reward',
           title='Final reward by training episode')
    ax.grid()

    plt.show()