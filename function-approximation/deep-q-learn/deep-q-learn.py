import gym
import time
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

RENDER_ENV = False

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        return batch

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

def pick_e_greedy_action(q_net, state, actions, epsilon):
    prob = np.random.rand()
    action = -1

    if prob < epsilon:
        action = np.random.choice(actions)
    else:
        with torch.no_grad():
            _, action = torch.max(q_net(state), 0)
            action = action.item()

    return action

# This is an inefficient implementation since the calculus could be done without the 'for' loop
# Still, it is easier to see the computation of the TD target and the loss in this way
def calculate_loss(device, exp_policy, target_policy, experiences, gamma, criterion_loss, double=True):
    loss = 0

    for state, action, reward, next_state, done in experiences:
        # Q_exp(s,a)
        q_value = exp_policy(state)[action]

        if done:
            # TD Target = r
            td_target = torch.tensor(reward).to(device, dtype=torch.float)
        else:
            if double:
                # TD Target = r + gamma * Q_target(s', arg_max Q_exp(s', .))
                next_action = torch.argmax(exp_policy(state))
                next_state_action_q_value = target_policy(next_state)[next_action]
            else:
                # TD Target = r + gamma * max Q_target(s',.)
                next_state_q_values = target_policy(next_state)
                next_state_action_q_value = torch.max(next_state_q_values)

            td_target = reward + gamma * next_state_action_q_value

        td_target = td_target.detach()

        # Calculate loss between guess and target
        loss = loss + criterion_loss(q_value, td_target)

    loss = loss / len(experiences)

    return loss

def q_learning(env, num_episodes, gamma, epsilon, learning_rate, buffer_size, batch_size):
    features_size = env.observation_space.shape[0]
    act_space_size = env.action_space.n
    env_actions = range(env.action_space.n)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # We need an e-greedy exploratory policy and a target policy
    exp_policy = Net(features_size, act_space_size).to(device)
    target_policy = Net(features_size, act_space_size).to(device)
    
    criterion_loss = nn.MSELoss()
    optimiser = optim.Adam(exp_policy.parameters(), lr=learning_rate)

    # Initialise replay buffer
    replay_buffer = ExperienceReplay(buffer_size)

    print('Filling replay buffer...')

    while len(replay_buffer) < buffer_size:
        finished = False

        # Initialize s
        obs = env.reset()
        obs = torch.from_numpy(obs).to(device, dtype=torch.float)

        while not finished:            
            # Chose a using policy derived from Q_exp (e-greedy)
            action = pick_e_greedy_action(exp_policy, obs, env_actions, epsilon)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            new_obs = torch.from_numpy(new_obs).to(device, dtype=torch.float)

            # Store experience in replay buffer
            replay_buffer.append((obs, action, reward, new_obs, finished))
            
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
        obs = torch.from_numpy(obs).to(device, dtype=torch.float)

        while not finished:
            if RENDER_ENV:
                env.render()
            
            # Chose a using policy derived from Q_exp (e-greedy)
            action = pick_e_greedy_action(exp_policy, obs, env_actions, epsilon)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            new_obs = torch.from_numpy(new_obs).to(device, dtype=torch.float)

            # Store experience in replay buffer
            replay_buffer.append((obs, action, reward, new_obs, finished))
            
            obs = new_obs
            ep_reward += reward

            # Sample experiences
            batch = replay_buffer.sample(batch_size)
            loss = calculate_loss(device, exp_policy, target_policy, batch, gamma, criterion_loss)
            
            # Optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

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
    env = gym.make('MountainCar-v0') #'MountainCar-v0' #'CartPole-v0'
    num_episodes = 2000
    gamma = 0.999
    epsilon = 0.8
    learning_rate = 0.0001
    buffer_size = 100000
    batch_size = 128

    start_time = time.time()
    q_net, rewards = q_learning(env, num_episodes, gamma, epsilon, learning_rate, buffer_size, batch_size)
    end_time = time.time()

    print('Q-Learning took', end_time - start_time, 'seconds')

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