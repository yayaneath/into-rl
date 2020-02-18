import gym
import numpy as np
import shadow_hand

MAX_STEPS = 20000

env = gym.make('shadowhand-v0')
features_size = env.observation_space.shape[0]

obs = env.reset()
done = False

env.render()

input("Press Enter to continue...")