import os
import gym
import math
import numpy as np
import pybullet as p
import pybullet_data
import datetime

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
    self.physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    self.seed()

    # self.reward_range could be defined, but by default it is [-inf, inf]

  # MANDATORY. Returns (obs, reward, done, info), info can be blank dictionary
  def step(self, action):
    self._env_step_counter += 1

    self._observation = [0.0, 0.0]
    reward = 0.0
    done = False

    # Check the state of the joints
    num_joints = p.getNumJoints(self.bot_id)

    for i in range(num_joints):
      state = p.getJointState(self.bot_id, i)

      print('\n', i, p.getJointInfo(self.bot_id, i)[1].decode('UTF-8'))
      print('position:', state[0])
      print('velocity:', state[1])
      print('forces:', state[2])
      print('torque:', state[3])
      # enableJointForceTorqueSensor in case forces are 0s

    # Check some link state
    link_id = 31
    some_link_state = p.getLinkState(self.bot_id, link_id, computeLinkVelocity=True)
    print('\nState of link', link_id)
    print('world position (CoM):', some_link_state[0]) # (CoM) Center of mass
    print('world orientation (CoM):', some_link_state[1])
    print('world position (frame):', some_link_state[4])
    print('world orientation (frame):', some_link_state[5])
    print('world linear velocity:', some_link_state[6])
    print('world angular velocity:', some_link_state[7])


    # Command a joint!
    joint_id = 30 # 30 is rh_THJ1 in range [0.0, pi/2]
    p.setJointMotorControl2(bodyUniqueId=self.bot_id,
                            jointIndex=joint_id,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=1.0,
                            # Use the max velocity from URDF
                            maxVelocity=p.getJointInfo(self.bot_id, joint_id)[11])

    p.stepSimulation()

    # Check some link state
    link_id = 31
    some_link_state = p.getLinkState(self.bot_id, link_id, computeLinkVelocity=True)
    print('\nState of link', link_id)
    print('world position (CoM):', some_link_state[0]) # (CoM) Center of mass
    print('world orientation (CoM):', some_link_state[1])
    print('world position (frame):', some_link_state[4])
    print('world orientation (frame):', some_link_state[5])
    print('world linear velocity:', some_link_state[6])
    print('world angular velocity:', some_link_state[7])

    return np.array(self._observation), reward, done, {}

  # MANDATORY.
  def reset(self):
    self._env_step_counter = 0

    p.resetSimulation()
    p.setGravity(0, 0, -9.8) # m/s^2
    p.setTimeStep(0.01) # sec

    plane_id = p.loadURDF('plane.urdf') # This one comes with bullet already

    # Set pose so the palm points upwards
    start_position = [0, 0, 0.2]
    start_orientation = p.getQuaternionFromEuler([-math.pi / 2.0, 0, 0])

    path = os.path.abspath(os.path.dirname(__file__)) # This file's path
    self.bot_id = p.loadURDF(os.path.join(path, 'shadowhand.urdf'), start_position,
                            start_orientation)

    # STATE_LOGGING_VIDEO_MP4 requires file to be '.mp4' and ffmpeg installed
    # STATE_LOGGING_GENERIC_ROBOT requires maxLogDof for robots with +12 DoF
    log_file_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, './logs/' + log_file_name,
                        objectUniqueIds=[self.bot_id],
                        maxLogDof=p.getNumJoints(self.bot_id))

    position, orientation = p.getBasePositionAndOrientation(self.bot_id)
    euler_orientation = p.getEulerFromQuaternion(orientation) # get pitch, roll, yaw
    num_joints = p.getNumJoints(self.bot_id)

    print('position:', position)
    print('euler_orientation:', euler_orientation)
    print('num_joints:', num_joints)

    for i in range(num_joints):
      info_joint = p.getJointInfo(self.bot_id, i)
    
      print('\njoint_id:', info_joint[0])
      print('name:', info_joint[1].decode('UTF-8'))
      print('type:', info_joint[2])
      print('lower:', info_joint[8])
      print('upper:', info_joint[9])
      print('child link name:', info_joint[12].decode('UTF-8'))
      print('parent link id:', info_joint[16])

    self._observation = [0.0, 0.0]

    return np.array(self._observation)

  # MANDATORY.
  def render(self, mode='human', close='False'):
    # Pybullet does everything for us, since we are using p.GUI
    pass

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)

    return [seed]