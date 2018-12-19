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

best_params = None
best_reward = 0
episodes = 10000
episodes_run = 0

for _ in range(episodes):
    parameters = np.random.rand(4) * 2 - 1
    total_reward = run_episode(env, parameters, MAX_TIME_STEPS)
    episodes_run += 1

    if total_reward > best_reward:
        best_params = parameters
        best_reward = total_reward

    # The agent keeps the pole standing for the whole episode
    if total_reward == MAX_TIME_STEPS:
        break

print('Solved in', episodes_run, 'episodes')