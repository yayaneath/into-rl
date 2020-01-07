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

# Two head network
class TwoHeadNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoHeadNet, self).__init__()
        self.fc1 = nn.Linear(input_size, int(input_size * 10))
        self.relu = nn.ReLU()
        self.action_head = nn.Linear(int(input_size * 10), output_size)
        self.value_head = nn.Linear(int(input_size * 10), 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        # Actor output
        action_probs = F.softmax(self.action_head(out), dim=0)

        # Critic output
        state_values = self.value_head(out)

        return action_probs, state_values

def pick_action(model, state):
    probs, value = model(state) # Get actions probabilities and state value
    cat_dist = Categorical(probs) # Create categorical distribution
    action = cat_dist.sample() # Sample an action

    # Return action, its log probability and state value for loss
    return action.item(), cat_dist.log_prob(action), value

def update_policy(model, optimiser, rewards, log_probs, state_values,
                  gamma, device):
    eps = 1e-8 # Avoid divison by zero
    step_return = 0.0 # Gt
    actor_loss = []
    critic_loss = []
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
        # Just the state value as item, not Torch tensor, so we disconnect
        # the graph of the actor from the critic. We only want to calculate
        # the loss for the actor using this state value.
        advantage = returns[i] - state_values[i].item()

        # Calculate actor loss
        actor_loss.append(-log_probs[i] * advantage)

        # Calculate critic loss: F.smooth_l1_loss, F.mse_loss...
        target_value = torch.tensor([returns[i]]).to(device)
        critic_loss.append(F.smooth_l1_loss(state_values[i], target_value))

    # Backward grads using sum of losses
    optimiser.zero_grad()
    final_loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
    final_loss.backward()
    optimiser.step()

def advantage_actor_critic(env, num_episodes, gamma, learning_rate):
    features_size = env.observation_space.shape[0]
    act_space_size = env.action_space.n

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TwoHeadNet(features_size, act_space_size).to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    rewards_evolution = []

    for ep in range(num_episodes + 1):
        finished = False
        ep_reward = 0.0
        rewards = []
        log_probs = []
        state_values = []

        # Initialize s
        obs = env.reset()
        obs = torch.from_numpy(obs).to(device, dtype=torch.float)

        while not finished:
            if RENDER_ENV:
                env.render()

            # Pick an action
            action, action_log_prob, state_value = pick_action(model, obs)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            # Save data for loss function
            log_probs.append(action_log_prob)
            state_values.append(state_value)
            rewards.append(reward)
            
            # Prepare next step
            obs = torch.from_numpy(new_obs).to(device, dtype=torch.float)
            ep_reward += reward

        rewards_evolution.append(ep_reward)

        # Now that the episode finished, calculate A2C update to actor and critic
        update_policy(model, optimiser, rewards, log_probs, state_values, 
                      gamma, device)

        if (ep % 1 == 0):
            print('Episode', ep, '- Total reward:', ep_reward)

    return model, rewards_evolution

if __name__ == '__main__':
    env = gym.make('CartPole-v0') #'MountainCar-v0' #'CartPole-v0'
    num_episodes = 2000
    gamma = 0.99
    learning_rate = 0.001

    start_time = time.time()
    actor, rewards = advantage_actor_critic(env, num_episodes, gamma, learning_rate)
    end_time = time.time()

    print('A2C took', end_time - start_time, 'seconds')

    file_name = 'advantage_actor_critic-' + str(np.mean(rewards)) + '-' + str(end_time)
    torch.save(actor.state_dict(), file_name)

    _, ax = plt.subplots()

    window_size = 25
    stride = 1
    window_avg = [ np.mean(rewards[i : i + window_size]) for i in range(0, len(rewards), stride) if i + window_size <= len(rewards) ]

    ax.plot(window_avg)
    ax.set(xlabel='episode (avg 25)', ylabel='total reward',
           title='Final reward by training episode')
    ax.grid()

    plt.show()