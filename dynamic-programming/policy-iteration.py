import gym
import time
import numpy as np

def policy_iteration(env, max_iterations, discount):
    values = np.zeros(env.observation_space.n)
    policy = np.random.randint(0, high=env.action_space.n, 
        size=env.observation_space.n)
    epsilon = 1e-20
    diff = 1.0
    count = 0

    while diff > epsilon:
        print('===== iteration', count, '=====')
        print(policy.reshape((4, 4)))

        values = policy_evaluation(env, values, policy, discount, max_iterations)
        new_policy = calculate_policy(env, values, discount)

        diff = np.mean(np.fabs(policy - new_policy))
        policy = new_policy

        count += 1

    return policy, values

def policy_evaluation(env, values, policy, discount, max_iter):
    states = np.arange(env.observation_space.n)
    changes = np.zeros(env.observation_space.n)
    epsilon = 1e-50

    for i in range(max_iter):
        for s in states:
            new_state_value = 0

            # instead of going through all of the actions, use the policy
            transitions = env.env.P[s][policy[s]]

            for j in range(len(transitions)):
                prob, next_s, reward, _ = transitions[j]
                new_state_value += prob * (reward + discount * values[next_s])

            changes[s] = abs(values[s] - new_state_value)
            values[s] = new_state_value
        
        if np.mean(changes) < epsilon:
            print('Policy evaluation converged after', i, 'iterations')
            break

    print(values.reshape((4, 4)))

    return values

def calculate_policy(env, values, discount):
    actions = np.arange(env.action_space.n)
    states = np.arange(env.observation_space.n)
    policy = np.zeros(env.observation_space.n).astype(int)

    for s in states:
        state_values = np.zeros(env.action_space.n)

        for a in actions:
            action_value = 0
            transitions = env.env.P[s][a]

            for j in range(len(transitions)):
                prob, next_s, reward, _ = transitions[j]
                action_value += prob * (reward + discount * values[next_s])
            
            state_values[a] = action_value
        
        # pi[s] = argmax_a(q(s,a))
        policy[s] = np.argmax(state_values)

    return policy

def run_env(env, policy, render=False):
    finished = False
    total_reward = 0

    obs = env.reset()

    while not finished:
        if render:
            env.render()

        obs, reward, finished, _ = env.step(policy[obs])
        total_reward += reward

    if render:
        env.render()

    return total_reward

def check_policy_performance(env, policy, max_episodes):
    scores = []
    
    for _ in range(max_episodes):
        scores.append(run_env(env, policy))

    return np.mean(scores)

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    max_iterations = 20000
    max_episodes = 1000
    gamma = 0.98

    start_time = time.time()
    policy, _ = policy_iteration(env, max_iterations, gamma)
    end_time = time.time()
    
    print('Policy iteration took', end_time - start_time, 'seconds')

    start_time = time.time()
    policy_score = check_policy_performance(env, policy, max_episodes)
    end_time = time.time()

    print('Policy performance check took', end_time - start_time, 'seconds')
    print('Average score:', policy_score)