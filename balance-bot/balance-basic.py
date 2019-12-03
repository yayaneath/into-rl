import gym
import numpy as np
import balance_bot

def run_episode(env, params):
    obs = env.reset()
    done = False

    total_reward = 0

    while not done:
        env.render()

        action = 0 if np.matmul(params, obs) < 0 else 1
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
    return total_reward

env = gym.make('balancebot-v0')
features_size = env.observation_space.shape[0]

noise_scale = 0.5
parameters = np.random.rand(features_size) * 2 - 1

best_reward = 0
episodes = 10000

for i in range(episodes):
    new_params = parameters + (np.random.rand(features_size) * 2 - 1) * noise_scale
    total_reward = run_episode(env, new_params)

    print(i, total_reward)

    if total_reward > best_reward:
        parameters = new_params
        best_reward = total_reward

        # The agent keeps the pole standing for the whole episode
        if total_reward > 150:
            print('Solved in', i, 'episodes')
            break