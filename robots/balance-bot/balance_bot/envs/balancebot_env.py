import os
import gym
import math
import numpy as np
import pybullet as p
import pybullet_data

from gym import error, spaces, utils
from gym.utils import seeding

# https://github.com/openai/gym/blob/master/gym/core.py
# This bot has a cube/body on two wheels which must be balanced upright
class BalancebotEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'],
              'video.frames_per_second': 50
             }

  def __init__(self):
    self._observation = []

    # Continuous actions
    self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,))
    self._min_speed = -5.0
    self._max_speed = 5.0
    
    # Continuous 3D space:
    #   pitch (inclination of cube)
    #   gyro (velocity of cube)
    #   commanded speed
    self.observation_space = spaces.Box(np.array([-math.pi, -math.pi, self._min_speed]),
                                        np.array([math.pi, math.pi, self._max_speed]))

    # Pybullet physics!
    # p.DIRECT or p.GUI
    # Use p.DIRECT in case of willing to create multiple simultaneous environments
    self.physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    self.seed()

    # self.reward_range should be defined, but by default it is [-inf, inf]

  # Returns (obs, reward, done, info), info can be blank dictionary
  def step(self, action):
    self._assign_throttle(action)
    p.stepSimulation()
    self._observation = self._compute_observation()
    reward = self._compute_reward()
    done = self._compute_done()

    self._envStepCounter += 1

    return np.array(self._observation), reward, done, {}

  def reset(self):
    self.vel = 0.0 # Init with 0 velocity
    self._envStepCounter = 0

    p.resetSimulation()
    p.setGravity(0, 0, -9.8) # m/s^2
    p.setTimeStep(0.01) # sec

    planeId = p.loadURDF('plane.urdf') # This one comes with bullet already
    cubeStartPos = [0, 0, 0.001]
    cubeStartOrn = p.getQuaternionFromEuler([0.0, 0, 0])

    path = os.path.abspath(os.path.dirname(__file__)) # This file's path
    self.botId = p.loadURDF(os.path.join(path, 'balancebot_simple.xml'),
                            cubeStartPos,
                            cubeStartOrn)

    self._observation = self._compute_observation() # Initial obs

    return np.array(self._observation)


  def render(self, mode='human', close='False'):
    # Pybullet does everything for us, since we are using p.GUI
    pass

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)

    return [seed]

  # Actions in Pybullet are asigned here
  def _assign_throttle(self, action):
    self.vel += action

    if self.vel > self._max_speed: self.vel = self._max_speed
    if self.vel < self._min_speed: self.vel = self._min_speed

    p.setJointMotorControl2(bodyUniqueId=self.botId,
                            jointIndex=0,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=self.vel)
    p.setJointMotorControl2(bodyUniqueId=self.botId,
                            jointIndex=1,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=-self.vel)

  def _compute_observation(self):
    _, cubeOrn = p.getBasePositionAndOrientation(self.botId)
    cubeEuler = p.getEulerFromQuaternion(cubeOrn) # get pitch, roll, yaw
    _, angular = p.getBaseVelocity(self.botId) # use only the angular velocity

    return [cubeEuler[0], angular[0], self.vel]

  def _compute_reward(self):
    _, cubeOrn = p.getBasePositionAndOrientation(self.botId)
    cubeEuler = p.getEulerFromQuaternion(cubeOrn) # get pitch, roll, yaw

    reward_cube = 1 - abs(cubeEuler[0]) # Pitch should be 0! (upright)
    reward_vel = abs(self.vel - 0.0) # Vel should be 0! (not moving)
    cube_weight = 0.1
    vel_weight = 0.01

    return reward_cube * cube_weight - reward_vel * vel_weight

  def _compute_done(self):
    cubePos, _ = p.getBasePositionAndOrientation(self.botId)

    # Cube is too low or we have run enough steps with the cube upright
    done = cubePos[2] < 0.15 or self._envStepCounter >= 200000

    return done

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This is the version with discrete action space

class BalancebotDiscEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'],
              'video.frames_per_second': 50
             }

  def __init__(self):
    self._observation = []

    # Discrete possible actions:
    #   specific velocity changes
    self._vel_change = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    self._min_speed = -5.0
    self._max_speed = 5.0
    self.action_space = spaces.Discrete(len(self._vel_change))
    
    self.observation_space = spaces.Box(np.array([-math.pi, -math.pi, self._min_speed]),
                                        np.array([math.pi, math.pi, self._max_speed]))
    
    # p.DIRECT or p.GUI
    # Use p.DIRECT in case of willing to create multiple simultaneous environments
    self.physicsClient = p.connect(p.GUI) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    self.seed()

  def step(self, action):
    self._assign_throttle(action)
    p.stepSimulation()
    self._observation = self._compute_observation()
    reward = self._compute_reward()
    done = self._compute_done()

    self._envStepCounter += 1

    return np.array(self._observation), reward, done, {}

  def reset(self):
    self.vel = 0.0
    self._envStepCounter = 0

    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(0.01)

    planeId = p.loadURDF('plane.urdf')
    cubeStartPos = [0, 0, 0.001]
    cubeStartOrn = p.getQuaternionFromEuler([0.0, 0, 0])

    path = os.path.abspath(os.path.dirname(__file__))
    self.botId = p.loadURDF(os.path.join(path, 'balancebot_simple.xml'),
                            cubeStartPos,
                            cubeStartOrn)

    self._observation = self._compute_observation()

    return np.array(self._observation)


  def render(self, mode='human', close='False'):
    pass

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)

    return [seed]

  def _assign_throttle(self, action):
    self.vel += self._vel_change[action]

    if self.vel > self._max_speed: self.vel = self._max_speed
    if self.vel < self._min_speed: self.vel = self._min_speed

    p.setJointMotorControl2(bodyUniqueId=self.botId,
                            jointIndex=0,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=self.vel)
    p.setJointMotorControl2(bodyUniqueId=self.botId,
                            jointIndex=1,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=-self.vel)

  def _compute_observation(self):
    _, cubeOrn = p.getBasePositionAndOrientation(self.botId)
    cubeEuler = p.getEulerFromQuaternion(cubeOrn)
    _, angular = p.getBaseVelocity(self.botId)

    return [cubeEuler[0], angular[0], self.vel]

  def _compute_reward(self):
    _, cubeOrn = p.getBasePositionAndOrientation(self.botId)
    cubeEuler = p.getEulerFromQuaternion(cubeOrn)

    reward_cube = 1 - abs(cubeEuler[0])
    reward_vel = abs(self.vel - 0.0)
    cube_weight = 0.1
    vel_weight = 0.01

    return reward_cube * cube_weight - reward_vel * vel_weight

  def _compute_done(self):
    cubePos, _ = p.getBasePositionAndOrientation(self.botId)

    done = cubePos[2] < 0.15 or self._envStepCounter >= 20000

    return done