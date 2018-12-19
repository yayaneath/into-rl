import gym

env = gym.make('CartPole-v0') # CartPole-v0, MsPacman-v0

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

for i_episode in range(1):
    obs = env.reset()

    for t in range(100):
        env.render()
        
        #print(obs)

        action = env.action_space.sample() # take a random action
        obs, reward, finished, info = env.step(action)

        #print('Reward:', reward)
        #print('Finished?', finished)
        #print('Info:', info)

        if finished:
            print('Episode finished after {} timesteps.'.format(t + 1))
            break