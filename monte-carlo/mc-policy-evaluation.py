import gym
import time
import numpy as np

def generate_episode_data(env, policy, render=False):
    finished = False
    states = []
    rewards = []

    obs = env.reset()

    while not finished:
        if render:
            env.render()

        states.append(obs)

        obs, reward, finished, _ = env.step(policy[obs])
        
        rewards.append(reward)

    return states, rewards


def mc_policy_evaluation(env, policy, num_episodes, gamma):
    state_visits = np.zeros(env.observation_space.n)
    state_returns = np.zeros(env.observation_space.n)
    values = np.zeros(env.observation_space.n)

    for ep in range(num_episodes):
        if (ep % 1000 == 0):
            print('=> Evaluating episode', ep)

        # Generate episode data
        ep_states, ep_rewards = generate_episode_data(env, policy)

        # Calculate values
        state_visited = np.zeros(env.observation_space.n)

        for i in range(len(ep_states)):
            state = ep_states[i]

            # If already visited, continue
            if state_visited[state]:
                continue

            return_G = 0.0

            # G_t = gamma ^ 0 * R_(t+1) + gamma ^ 1 * R_(t+2) + ... gamma ^ (T-1) * R_(t+T)
            for j in range(i, len(ep_states)):
                return_G += (gamma ** (j - i)) * ep_rewards[j]

            # N(s) = N(s) + 1
            state_visits[state] += 1
            # S(s) = S(s) + G_t
            state_returns[state] += return_G 

            # V(s) = S(s) / N(s)
            values[state] = state_returns[state] / state_visits[state]

    return values


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    num_episodes = 50000
    gamma = 0.98
    policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0] # Obtained from policy iteration
    
    start_time = time.time()
    policy_values = mc_policy_evaluation(env, policy, num_episodes, gamma)
    end_time = time.time()

    print('Policy evaluation took', end_time - start_time, 'seconds')
    print('Values:\n', policy_values.reshape((4, 4)))