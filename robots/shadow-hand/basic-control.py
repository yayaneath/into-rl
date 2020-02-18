import gym
import numpy as np
import shadow_hand

MAX_STEPS = 20000

env = gym.make('shadowhand-v0')

print('action_space:', env.action_space)
print(env.action_space.high, env.action_space.low)
print('observation_space:', env.observation_space)
print(env.observation_space.high, env.observation_space.low)

obs = env.reset()
done = False

for i in range(100000000):
	env.render()

	env.step(0)

	input('type')