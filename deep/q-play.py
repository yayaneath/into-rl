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
        self.fc1 = nn.Linear(input_size, int(input_size / 2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(input_size / 2), output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out

if __name__ == '__main__':
    #model_file = sys.argv[1]
    model_file = 'qnet-27-02-19'

    env = gym.make('CartPole-v0') #('MountainCar-v0')

    obs_space_size = env.observation_space.shape[0]
    act_space_size = env.action_space.n

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Q-net
    policy = Net(obs_space_size, act_space_size).to(device)
    policy.load_state_dict(torch.load(model_file))
    policy.eval()

    # Initialize s
    obs = env.reset()

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

        ep_reward += reward
        steps += 1

    print('Reward:', ep_reward)