import gym
import time
import numpy as np

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

    # This could be useful for normalising the observations
    #print(env.observation_space.high)
    #print(env.observation_space.low)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net(obs_space_size, act_space_size).to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for ep in range(num_episodes):
        if (ep % 1 == 0):
            print('=> Evaluating episode', ep)

        finished = False
        ep_reward = 0.0

        # Initialize s
        obs = env.reset()
        obs = torch.from_numpy(obs).to(device, dtype=torch.float)

        while not finished:
            if RENDER_ENV:
                env.render()
            
            # Chose a using policy derived from Q (e-greedy)
            action = pick_e_greedy_action(net, obs, env_actions, epsilon)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            # Set grads to zero
            optimizer.zero_grad()

            # Q(s,.)
            q_values = net(obs)[action]

            # TD Target = r + gamma * max Q(s',.)
            new_obs = torch.from_numpy(new_obs).to(device, dtype=torch.float)

            q_values_next = net(new_obs)
            td_target = reward + gamma * torch.max(q_values_next)

            # Calculate loss between guess and target
            loss = mse_loss(q_values, td_target)
            
            # Optimize
            loss.backward()
            optimizer.step()

            obs = new_obs
            ep_reward += reward
        
        if (ep % 1 == 0):
            print('=> Total reward:', ep_reward)

        epsilon -= 0.0001

    return net

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    num_episodes = 50000
    gamma = 0.98
    epsilon = 1.0
    learning_rate = 0.01

    start_time = time.time()
    q_net = q_learning(env, num_episodes, gamma, epsilon, learning_rate)
    end_time = time.time()

    print('Q-Learning (linear approx) took', end_time - start_time, 'seconds')