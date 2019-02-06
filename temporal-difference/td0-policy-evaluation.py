import gym
import time
import numpy as np

def td0_policy_evaluation(env, policy, num_episodes, gamma, alpha):
    values = np.zeros(env.observation_space.n)

    for ep in range(num_episodes):
        if (ep % 1000 == 0):
            print('=> Evaluating episode', ep)

        finished = False
        obs = env.reset()

        while not finished:
            new_obs, reward, finished, _ = env.step(policy[obs])

            # V(S) <- V(S) + alpha[R + gamma * V(S') - V(S)]
            values[obs] = values[obs] + alpha * (reward + gamma * values[new_obs] - values[obs])

            obs = new_obs

    return values


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    num_episodes = 50000
    gamma = 0.98
    alpha = 0.1
    policy = [0, 3, 3, 3, 
              0, 0, 0, 0,
              3, 1, 0, 0,
              0, 2, 1, 0] # Obtained from policy iteration
    
    start_time = time.time()
    policy_values = td0_policy_evaluation(env, policy, num_episodes, gamma, alpha)
    end_time = time.time()

    print('Policy evaluation took', end_time - start_time, 'seconds')
    print('Values:\n', policy_values.reshape((4, 4)))