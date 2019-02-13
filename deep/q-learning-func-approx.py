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

    # This could be useful for normalising the observations
    #print(env.observation_space.high)
    #print(env.observation_space.low)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # We need an e-greedy exploratory policy and a target policy
    exp_policy = Net(obs_space_size, act_space_size).to(device)
    target_policy = Net(obs_space_size, act_space_size).to(device)
    
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(exp_policy.parameters(), lr=learning_rate)

    rewards = []

    for ep in range(num_episodes):
        # Update the target policy every X episores
        if (ep % 10 == 0):
            print('=> Evaluating episode', ep)
            target_policy.load_state_dict(exp_policy.state_dict())

        finished = False
        ep_reward = 0.0

        # Initialize s
        obs = env.reset()
        obs = torch.from_numpy(obs).to(device, dtype=torch.float)

        while not finished:
            if RENDER_ENV:
                env.render()
            
            # Chose a using policy derived from Q (e-greedy)
            action = pick_e_greedy_action(exp_policy, obs, env_actions, epsilon)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            # Set grads to zero
            optimizer.zero_grad()

            # Q(s,a)
            q_value = exp_policy(obs)[action]

            # TD Target = r + gamma * max Q(s',.)
            new_obs = torch.from_numpy(new_obs).to(device, dtype=torch.float)

            next_state_q_values = target_policy(new_obs)
            td_target = reward + gamma * torch.max(next_state_q_values)

            # Calculate loss between guess and target
            loss = mse_loss(q_value, td_target)
            
            # Optimize
            loss.backward()
            optimizer.step()

            obs = new_obs
            ep_reward += reward
        
        if (ep % 10 == 0):
            print('Total reward:', ep_reward)
            print('Epsilon:', epsilon)

        epsilon -= 0.00002
        rewards.append(ep_reward)

    return td_target, rewards

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

    ax.plot(rewards)
    ax.set(xlabel='episode', ylabel='total reward',
           title='Final reward by training episode')
    ax.grid()

    plt.show()