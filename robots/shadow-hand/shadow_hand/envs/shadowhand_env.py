import os
import gym
import math
import numpy as np
import pybullet as p
import pybullet_data

from gym import error, spaces, utils
from gym.utils import seeding

# https://github.com/openai/gym/blob/master/gym/core.py

class ShadowhandEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'],
              'video.frames_per_second': 50
             }

  def __init__(self):
    self._observation = []

    # Continuous actions for joint robot0:WRJ1
    self.action_space = spaces.Box(low=-0.489, high=0.14, shape=(1,))
    
    # Continuous 3D space:
    #   pitch (inclination of cube)
    #   gyro (velocity of cube)
    self.observation_space = spaces.Box(np.array([-math.pi, -math.pi]),
                                        np.array([math.pi, math.pi]))

    # Pybullet physics!
    # p.DIRECT or p.GUI
    # Use p.DIRECT in case of willing to create multiple simultaneous environments
    self.physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    self.seed()

    # self.reward_range should be defined, but by default it is [-inf, inf]

  # MANDATORY. Returns (obs, reward, done, info), info can be blank dictionary
  def step(self, action):
    p.stepSimulation()

    self._envStepCounter += 1

    self._observation = [0.0, 0.0]
    reward = 0.0
    done = False

    return np.array(self._observation), reward, done, {}

  # MANDATORY.
  def reset(self):
    self._envStepCounter = 0

    p.resetSimulation()
    p.setGravity(0, 0, -9.8) # m/s^2
    p.setTimeStep(0.01) # sec

    planeId = p.loadURDF('plane.urdf') # This one comes with bullet already

    path = os.path.abspath(os.path.dirname(__file__)) # This file's path
    self.botId = p.loadURDF(os.path.join(path, 'shadowhand.urdf'))

    self._observation = [0.0, 0.0]

    return np.array(self._observation)

  # MANDATORY.
  def render(self, mode='human', close='False'):
    # Pybullet does everything for us, since we are using p.GUI
    pass

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)

    return [seed]