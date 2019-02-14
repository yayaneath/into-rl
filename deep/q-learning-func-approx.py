import gym
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

RENDER_ENV = True

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc1(x)

        return x

def pick_e_greedy_action(q_net, state, actions, epsilon):
    prob = np.random.rand()
    action = -1

    if prob < epsilon:
        action = np.random.choice(actions)
    else:
        _, action = torch.max(q_net(state), 0)
        action = action.item()

    return action

def q_learning(env, num_episodes, gamma, epsilon, learning_rate):
    obs_space_size = env.observation_space.shape[0]
    act_space_size = env.action_space.n
    env_actions = range(env.action_space.n)

    # Needed to scale features
    min_pos, min_vel = env.observation_space.low
    max_pos, max_vel = env.observation_space.high

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # We need an e-greedy exploratory policy and a target policy
    exp_policy = Net(obs_space_size, act_space_size).to(device)
    target_policy = Net(obs_space_size, act_space_size).to(device)
    
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(exp_policy.parameters(), lr=learning_rate)

    rewards = []
    losses = []

    for ep in range(num_episodes):
        # Update the target policy every X episores
        if (ep % 1 == 0):
            print('=> Evaluating episode', ep)

        if (ep % 10 == 0):
            target_policy.load_state_dict(exp_policy.state_dict())

        finished = False
        ep_reward = 0.0
        ep_loss = []

        # Initialize s
        obs_pos, obs_vel = env.reset()
        scaled_obs_pos = (obs_pos - min_pos) / (max_pos - min_pos)
        scaled_obs_vel = (obs_vel - min_vel) / (max_vel - min_vel)
        obs = np.array([scaled_obs_pos, scaled_obs_vel])

        obs = torch.from_numpy(obs).to(device, dtype=torch.float)

        while not finished:
            if RENDER_ENV:
                env.render()
            
            # Chose a using policy derived from Q (e-greedy)
            action = pick_e_greedy_action(exp_policy, obs, env_actions, epsilon)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            new_obs_pos, new_obs_vel = new_obs
            scaled_new_obs_pos = (new_obs_pos - min_pos) / (max_pos - min_pos)
            scaled_new_obs_vel = (new_obs_vel - min_vel) / (max_vel - min_vel)
            new_obs = np.array([scaled_new_obs_pos, scaled_new_obs_vel])

            # Set grads to zero
            optimizer.zero_grad()

            # Q(s,a)
            q_value = exp_policy(obs)[action]

            # TD Target = r + gamma * max Q(s',.)
            new_obs = torch.from_numpy(new_obs).to(device, dtype=torch.float)

            next_state_q_values = target_policy(new_obs)
            next_state_q_values = next_state_q_values.detach()
            td_target = reward + gamma * torch.max(next_state_q_values)

            if finished:
                td_target = torch.tensor(reward).to(device, dtype=torch.float)

            # Calculate loss between guess and target
            loss = mse_loss(q_value, td_target)
            
            # Optimize
            loss.backward()
            optimizer.step()

            obs = new_obs
            ep_reward += reward
            ep_loss.append(loss.item())

        avg_loss = np.mean(ep_loss)
        
        if (ep % 1 == 0):
            print('Total reward:', ep_reward)
            print('Avg Loss:', avg_loss)
            print('Epsilon:', epsilon)

        epsilon -= 0.00002
        rewards.append(ep_reward)
        losses.append(avg_loss)

    return target_policy, rewards

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    num_episodes = 10000
    gamma = 0.98
    epsilon = 0.2
    learning_rate = 0.01

    start_time = time.time()
    q_net, rewards = q_learning(env, num_episodes, gamma, epsilon, learning_rate)
    end_time = time.time()

    print('Q-Learning (linear approx) took', end_time - start_time, 'seconds')

    file_name = 'q-net-' + str(np.mean(rewards)) + '-' + str(end_time)
    torch.save(q_net.state_dict(), file_name)

    _, ax = plt.subplots()

    # Plot an average reward within a window of 25 episodes?
    ax.plot(rewards)
    ax.set(xlabel='episode', ylabel='total reward',
           title='Final reward by training episode')
    ax.grid()

    plt.show()