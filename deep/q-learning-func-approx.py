import gym
import time
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

RENDER_ENV = False

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


def featurise_state(scaler, featurizer, state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    
    return featurized[0]

def q_learning(env, num_episodes, gamma, epsilon, learning_rate):
    #obs_space_size = env.observation_space.shape[0]
    act_space_size = env.action_space.n
    env_actions = range(env.action_space.n)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Feature Preprocessing: Normalise to zero mean and unit variance
    # We use a few samples from the observation space to do this
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # Used to convert a state to a featurises represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    features_size = 400
    featuriser = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
            ])
    featuriser.fit(scaler.transform(observation_examples))

    # We need an e-greedy exploratory policy and a target policy
    exp_policy = Net(features_size, act_space_size).to(device)
    target_policy = Net(features_size, act_space_size).to(device)
    
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
        obs = env.reset()
        obs_feat = featurise_state(scaler, featuriser, obs)
        obs = torch.from_numpy(obs_feat).to(device, dtype=torch.float)

        while not finished:
            if RENDER_ENV:
                env.render()
            
            # Chose a using policy derived from Q (e-greedy)
            action = pick_e_greedy_action(exp_policy, obs, env_actions, epsilon)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            obs_feat = featurise_state(scaler, featuriser, new_obs)
            new_obs = torch.from_numpy(obs_feat).to(device, dtype=torch.float)

            # Set grads to zero
            optimizer.zero_grad()

            # Q(s,a)
            q_value = exp_policy(obs)[action]

            # TD Target = r + gamma * max Q(s',.)
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
    num_episodes = 5000
    gamma = 0.98
    epsilon = 0.6
    learning_rate = 0.01

    start_time = time.time()
    q_net, rewards = q_learning(env, num_episodes, gamma, epsilon, learning_rate)
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