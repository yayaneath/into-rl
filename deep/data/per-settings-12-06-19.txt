import gym
import time
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

import sklearn.preprocessing
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

RENDER_ENV = False

class NaivePrioritizedBuffer:
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for experience in samples:
            states.append(np.concatenate(experience[0]))
            actions.append(experience[1])
            rewards.append(experience[2])
            next_states.append(np.concatenate(experience[3]))
            dones.append(experience[4])
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), \
            np.array(dones, dtype=np.float32), indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

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

        return out

def pick_e_greedy_action(device, q_net, state, actions, epsilon):
    prob = np.random.rand()
    action = -1

    if prob < epsilon:
        action = np.random.choice(actions)
    else:
        state = torch.from_numpy(state).to(device, dtype=torch.float)
        
        with torch.no_grad():
            _, action = torch.max(q_net(state), 0)
            action = action.item()

    return action

def compute_td_loss(device, optimiser, exp_policy, target_policy, replay_buffer, batch_size, gamma, beta):
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta) 

    state = torch.from_numpy(state).to(device, dtype=torch.float)
    next_state = torch.from_numpy(next_state).to(device, dtype=torch.float)
    action = torch.from_numpy(action).to(device, dtype=torch.long)
    reward = torch.from_numpy(reward).to(device, dtype=torch.float)
    done = torch.from_numpy(done).to(device, dtype=torch.float)
    weights = torch.from_numpy(weights).to(device, dtype=torch.float)

    q_values = exp_policy(state)
    next_q_values = exp_policy(next_state)
    next_q_state_values = target_policy(next_state) 

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()
        
    optimiser.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimiser.step()

def q_learning(env, num_episodes, gamma, epsilon, learning_rate, buffer_size, batch_size, beta):
    features_size = env.observation_space.shape[0]
    act_space_size = env.action_space.n
    env_actions = range(env.action_space.n)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # We need an e-greedy exploratory policy and a target policy
    exp_policy = Net(features_size, act_space_size).to(device)
    target_policy = Net(features_size, act_space_size).to(device)

    optimiser = optim.Adam(exp_policy.parameters(), lr=learning_rate)

    # Initialise replay buffer
    replay_buffer = NaivePrioritizedBuffer(buffer_size)

    print('Filling replay buffer...')

    while len(replay_buffer) < buffer_size:
        finished = False

        # Initialize s
        obs = env.reset()

        while not finished:            
            # Chose a using policy derived from Q_exp (e-greedy)
            action = pick_e_greedy_action(device, exp_policy, obs, env_actions, epsilon)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            # Store experience in replay buffer
            replay_buffer.push(obs, action, reward, new_obs, finished)
            
            obs = new_obs

    # Start training
    rewards = []

    for ep in range(num_episodes + 1):
        if (ep % 1 == 0):
            print('=> Evaluating episode', ep)

        finished = False
        ep_reward = 0.0

        # Initialize s
        obs = env.reset()

        while not finished:
            if RENDER_ENV:
                env.render()
            
            # Chose a using policy derived from Q_exp (e-greedy)
            action = pick_e_greedy_action(device, exp_policy, obs, env_actions, epsilon)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            # Store experience in replay buffer
            replay_buffer.push(obs, action, reward, new_obs, finished)
            
            obs = new_obs
            ep_reward += reward

            # Optimise
            compute_td_loss(device, optimiser, exp_policy, target_policy, replay_buffer, batch_size, gamma, beta)

        if (ep % 1 == 0):
            print('Total reward:', ep_reward)
            print('Epsilon:', epsilon)

        # Update the target policy every X episodes
        if (ep % 250 == 0):
            target_policy.load_state_dict(exp_policy.state_dict())
            target_policy.eval()

        epsilon -= 0.0004
        rewards.append(ep_reward)

    target_policy.load_state_dict(exp_policy.state_dict())
    target_policy.eval()

    return target_policy, rewards

if __name__ == '__main__':
    env = gym.make('CartPole-v0') #('MountainCar-v0')
    num_episodes = 2000
    gamma = 0.99
    epsilon = 0.8
    learning_rate = 0.0001
    buffer_size = 100000
    batch_size = 128
    beta = 0.6

    start_time = time.time()
    q_net, rewards = q_learning(env, num_episodes, gamma, epsilon, learning_rate, buffer_size, batch_size, beta)
    end_time = time.time()

    print('Q-Learning (linear approx) took', end_time - start_time, 'seconds')

    file_name = 'q-net-' + str(np.mean(rewards)) + '-' + str(end_time)
    torch.save(q_net.state_dict(), file_name)

    _, ax = plt.subplots()

    window_size = 25
    stride = 1
    window_avg = [ np.mean(rewards[i : i + window_size]) for i in range(0, len(rewards), stride) if i + window_size <= len(rewards) ]

    ax.plot(window_avg)
    ax.set(xlabel='episode (avg 25)', ylabel='total reward',
           title='Final reward by training episode')
    ax.grid()

    plt.show()