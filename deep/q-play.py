import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

RENDER_ENV = True

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc1(x)

        return x

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    obs_space_size = env.observation_space.shape[0]
    act_space_size = env.action_space.n

    min_pos, min_vel = env.observation_space.low
    max_pos, max_vel = env.observation_space.high

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Q-net
    policy = Net(obs_space_size, act_space_size).to(device)
    policy.load_state_dict(torch.load(sys.argv[1]))
    policy.eval()

    # Initialize s
    obs_pos, obs_vel = env.reset()
    scaled_obs_pos = (obs_pos - min_pos) / (max_pos - min_pos)
    scaled_obs_vel = (obs_vel - min_vel) / (max_vel - min_vel)
    obs = np.array([scaled_obs_pos, scaled_obs_vel])


    ep_reward = 0.0
    finished = False
    steps = 0

    while not finished:
        print('Playing step...', steps)

        if RENDER_ENV:
            env.render()
    
        obs = torch.from_numpy(obs).to(device, dtype=torch.float)
        
        # Chose a using policy
        _, action = torch.max(policy(obs), 0)
        action = action.item()

        # Take action a and observe r, s'
        obs, reward, finished, _ = env.step(action)

        obs_pos, obs_vel = obs
        scaled_obs_pos = (obs_pos - min_pos) / (max_pos - min_pos)
        scaled_obs_vel = (obs_vel - min_vel) / (max_vel - min_vel)
        obs = np.array([scaled_obs_pos, scaled_obs_vel])

        ep_reward += reward
        steps += 1

    print('Reward:', ep_reward)