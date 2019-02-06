import gym
import time
import numpy as np

def pick_e_greedy_action(q_values, state, actions, epsilon):
    prob = np.random.rand()
    action = -1

    if prob < epsilon:
        action = np.random.choice(actions)
    else:
        action = np.argmax(q_values[state])

    return action

def q_learning(env, num_episodes, gamma, alpha, epsilon):
    q_values = np.zeros((env.observation_space.n, env.action_space.n))
    env_actions = range(env.action_space.n)

    for ep in range(num_episodes):
        if (ep % 1000 == 0):
            print('=> Evaluating episode', ep)

        finished = False

        # Initialize s
        obs = env.reset()

        while not finished:
            # Chose a using policy derived from Q (e-greedy)
            action = pick_e_greedy_action(q_values, obs, env_actions, epsilon)

            # Take action a and observe r, s'
            new_obs, reward, finished, _ = env.step(action)

            # Q(s,a) <- Q(s,a) + alpha[R + gamma * max Q(s') - Q(s,a)]
            q_values[obs][action] = q_values[obs][action] + alpha * (reward + gamma * np.max(q_values[new_obs]) - q_values[obs][action])

            obs = new_obs

    # Caculate V from Q
    values = np.max(q_values, axis=1)
    print(values.reshape(4, 4))

    # Generate policy pi(s) = argmax Q(s,a)
    policy = np.argmax(q_values, axis=1)
    print(policy.reshape(4, 4))

    return policy

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    num_episodes = 50000
    gamma = 0.98
    alpha = 0.1
    epsilon = 0.2
    
    start_time = time.time()
    policy_values = q_learning(env, num_episodes, gamma, alpha, epsilon)
    end_time = time.time()

    print('Q-Learning took', end_time - start_time, 'seconds')