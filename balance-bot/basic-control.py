import gym
import numpy as np
import balance_bot

MAX_STEPS = 20000

def run_episode(env, params):
    obs = env.reset()
    done = False

    total_reward = 0

    for i in range(MAX_STEPS):
        env.render()

        action = 0 if np.matmul(params, obs) < 0 else 1
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break
        
        print(i)

    return total_reward

env = gym.make('balancebotdisc-v0')
features_size = env.observation_space.shape[0]

noise_scale = 0.5
parameters = np.random.rand(features_size) * 2 - 1

best_reward = 0
best_episode = 0
episodes = 100

for i in range(episodes):
    new_params = parameters + (np.random.rand(features_size) * 2 - 1) * noise_scale
    total_reward = run_episode(env, new_params)

    print(i, total_reward)

    if total_reward > best_reward:
        parameters = new_params
        best_reward = total_reward
        best_episode = i

print('Solved in', best_episode, 'episodes with reward', best_reward)