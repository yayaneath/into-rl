import gym
import time
import numpy as np

def pick_e_greedy_action(policy, state, actions, epsilon):
    prob = np.random.rand()
    action = -1

    if prob < epsilon:
        action = np.random.choice(actions)
    else:
        action = policy[state]

    return action


def generate_episode_data(env, policy, epsilon, render=False):
    finished = False
    states_actions = []
    rewards = []
    env_actions = range(env.action_space.n)

    obs = env.reset()

    while not finished:
        if render:
            env.render()

        action = pick_e_greedy_action(policy, obs, env_actions, epsilon)
        states_actions.append((obs, action))

        obs, reward, finished, _ = env.step(action)
        
        rewards.append(reward)

    return states_actions, rewards

def on_policy_mc_control(num_episodes, env, epsilon, gamma):
    states_actions_visits = np.zeros((env.observation_space.n, env.action_space.n))
    states_actions_returns = np.zeros((env.observation_space.n, env.action_space.n))
    q_values = np.zeros((env.observation_space.n, env.action_space.n))
    policy = np.random.randint(0, high=env.action_space.n, 
        size=env.observation_space.n)

    for ep in range(num_episodes):
        if (ep % 1000 == 0):
            print('=> Evaluating episode', ep)

        # Generate episode data
        ep_states_actions, ep_rewards = generate_episode_data(env, policy, epsilon)
        
        # Calculate values
        states_actions_seen = np.zeros((env.observation_space.n, env.action_space.n))

        for i in range(len(ep_states_actions)):
            state, action = ep_states_actions[i]

            # If already visited, continue
            if states_actions_seen[state][action]:
                continue

            return_G = 0.0

            # G_t = gamma ^ 0 * R_(t+1) + gamma ^ 1 * R_(t+2) + ... gamma ^ (T-1) * R_(t+T)
            for j in range(i, len(ep_states_actions)):
                return_G += (gamma ** (j - i)) * ep_rewards[j]

            # N(s,a) = N(s,a) + 1
            states_actions_visits[state][action] += 1
            # S(s,a) = S(s,a) + G_t
            states_actions_returns[state][action] += return_G 

            # Q(s,a) = S(s,a) / N(s,a)
            q_values[state][action] = states_actions_returns[state][action] / states_actions_visits[state][action]

            # Improve Policy
            policy[state] = np.argmax(q_values[state])

    values = np.max(q_values, axis=1)
    print(values.reshape(4, 4))

    # Change actions at those state that are never visited or have 0.0 value
    policy = np.argmax(q_values, axis=1)

    return policy

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    num_episodes = 50000
    gamma = 0.98
    epsilon = 0.1
    
    start_time = time.time()
    policy_values = on_policy_mc_control(num_episodes, env, epsilon, gamma)
    end_time = time.time()

    print('Policy control', end_time - start_time, 'seconds')
    print('Values:\n', policy_values.reshape((4, 4)))