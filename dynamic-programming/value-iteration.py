import gym
import time
import numpy as np

def value_iteration(env, max_iter, discount):
    actions = np.arange(env.action_space.n)
    states = np.arange(env.observation_space.n)
    values_old = np.zeros(env.observation_space.n)
    values_new = np.copy(values_old)
    epsilon = 1e-50

    for i in range(max_iter):
        values_old = np.copy(values_new)

        # for each state s in S
        for s in states:
            state_values = []

            # for all actions a in A
            for a in actions:
                action_value = 0

                # tuples of (prob, next_state, reward, done) for each resulting transition
                # if stochastic, there are more than one
                # otherwise (deterministic), there is one
                transitions = env.env.P[s][a]

                # for each s' in S, coming from s through action a
                for j in range(len(transitions)):
                    prob, next_s, reward, _ = transitions[j]

                    # here, the reward is linked to the triplet <s, a, s'>
                    # in some other cases it is only linked to the tuple <s, a>
                    # the only difference is notation and numerically the resulting values change
                    # but the logic is the same (the resulting policy that can be expressed is identical)

                    # p(s, a, s') * [r(s, a, s') + gamma * v(s')]
                    action_value += prob * (reward + discount * values_old[next_s])
                
                state_values.append(action_value)
            
            # max_a(...)
            values_new[s] = max(state_values)

        if np.mean(np.fabs(values_new - values_old)) < epsilon:
            print('Value Iteration converged after', i, 'iterations')
            break

    return values_new

def inplace_value_iteration(env, max_iter, discount):
    actions = np.arange(env.action_space.n)
    states = np.arange(env.observation_space.n)
    values = np.zeros(env.observation_space.n)
    changes = np.zeros(env.observation_space.n)
    epsilon = 1e-50

    for i in range(max_iter):
        for s in states:
            state_values = []

            for a in actions:
                action_value = 0
                transitions = env.env.P[s][a]

                for j in range(len(transitions)):
                    prob, next_s, reward, _ = transitions[j]
                    action_value += prob * (reward + discount * values[next_s])
                
                state_values.append(action_value)
            
            changes[s] = abs(values[s] - max(state_values))
            values[s] = max(state_values)

        if np.mean(changes) < epsilon:
            print('Value Iteration converged after', i, 'iterations')
            break

    return values

def calculate_policy(env, values, discount):
    actions = np.arange(env.action_space.n)
    states = np.arange(env.observation_space.n)
    policy = np.zeros(env.observation_space.n)

    for s in states:
        state_values = np.zeros(env.action_space.n)

        for a in actions:
            action_value = 0
            transitions = env.env.P[s][a]

            for j in range(len(transitions)):
                prob, next_s, reward, _ = transitions[j]
                action_value += prob * (reward + discount * values[next_s])
            
            state_values[a] = action_value
        
        # pi_*[s] = argmax_a(q_*(s,a))
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
    max_iterations = 10000
    max_episodes = 1000
    gamma = 0.98 # Discount factor

    start_time = time.time()
    values = value_iteration(env, max_iterations, gamma)
    end_time = time.time()
    value_iteration_time = end_time - start_time

    print('Value iteration took', value_iteration_time, 'seconds')
    print(values.reshape((4, 4)))

    start_time = time.time()
    policy = calculate_policy(env, values, gamma).astype(int)
    end_time = time.time()

    print('Policy calculation took', end_time - start_time, 'seconds')
    print(policy.reshape((4, 4)))

    start_time = time.time()
    policy_score = check_policy_performance(env, policy, max_episodes)
    end_time = time.time()

    print('Policy performance check took', end_time - start_time, 'seconds')
    print('Average score:', policy_score)

    print('\n====================\n')
    #print('Playing a game!')
    #run_env(env, policy, render=True)

    start_time = time.time()
    inplace_values = inplace_value_iteration(env, max_iterations, gamma)
    end_time = time.time()
    inplace_value_iteration_time = end_time - start_time

    print('In-place value iteration took', inplace_value_iteration_time, 'seconds')
    print(inplace_values.reshape((4, 4)))

    print('Difference between values from value iteration and in-place method:')
    print(values - inplace_values)
    print('Time difference:', inplace_value_iteration_time - value_iteration_time,
        'seconds (improvement of', 
        round(100 - (inplace_value_iteration_time * 100) / value_iteration_time, 2), '%)')