import gym
import numpy as np
import balance_bot

# A basic P controller can balance the robot

env = gym.make('balancebot-v0')
obs = env.reset()

done = False
total_reward = 0

kp = 20
error = 0
target = 0
vel = 0.0

while not done:
    env.render()

    error = target - obs[0]
    vel = error * kp

    obs, reward, done, _ = env.step(vel)
    total_reward += reward

    print(total_reward)