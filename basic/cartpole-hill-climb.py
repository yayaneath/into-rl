import gym
import numpy as np

MAX_TIME_STEPS = 200

def run_episode(env, params, steps):
    obs = env.reset()

    total_reward = 0

    for _ in range(steps):
        env.render()

        action = 0 if np.matmul(params, obs) < 0 else 1
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break
        
    return total_reward

env = gym.make('CartPole-v0')

noise_scale = 0.5
parameters = np.random.rand(4) * 2 - 1

best_reward = 0
episodes = 10000

for i in range(episodes):
    new_params = parameters + (np.random.rand(4) * 2 - 1) * noise_scale
    total_reward = run_episode(env, new_params, MAX_TIME_STEPS)

    if total_reward > best_reward:
        parameters = new_params
        best_reward = total_reward

        # The agent keeps the pole standing for the whole episode
        if total_reward == MAX_TIME_STEPS:
            print('Solved in', i, 'episodes')
            break