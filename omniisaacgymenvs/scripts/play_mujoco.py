import mujoco
#@title Import packages for plotting and creating graphics
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

# Graphics and plotting.
import matplotlib.pyplot as plt


from gym import spaces
import numpy as np
import torch
import yaml

from rl_games.algos_torch.players import BasicPpoPlayerContinuous, BasicPpoPlayerDiscrete




config_name = "/home/antoine/Documents/Omniverse/omniisaacgymenvs/cfg/train/MFP2D_PPOmulti_dict_MLP.yaml"
#model_name = "/home/antoine/Documents/Omniverse/omniisaacgymenvs/runs/MFP2D_Virtual_GoToXY/nn/last_MFP2D_Virtual_GoToXY_ep_1000_rew__607.5902_.pth"
model_name = "/home/antoine/Documents/Omniverse/penalty/MLP_GTXY_UF_0.25_ST_PE_0.03_PAV_1.5_PLV_0.01/nn/last_MLP_GTXY_UF_0.25_ST_PE_0.03_PAV_1.5_PLV_0.01_ep_2000_rew__555.7034_.pth"

class RLGamesModel:
    def __init__(self):
        self.obs = dict({'state':torch.zeros((1,10), dtype=torch.float32, device='cuda'),
                    'transforms': torch.zeros(5,8, device='cuda'),
                    'masks': torch.zeros(8, dtype=torch.float32, device='cuda')})

    def buildModel(self):
        act_space = spaces.Tuple([spaces.Discrete(2)]*8)
        obs_space = spaces.Dict({"state":spaces.Box(np.ones(10) * -np.Inf, np.ones(10) * np.Inf),
                                 "transforms":spaces.Box(low=-1, high=1, shape=(8, 5)),
                                 "masks":spaces.Box(low=0, high=1, shape=(8,))})
        self.player = BasicPpoPlayerDiscrete(self.cfg, obs_space, act_space, clip_actions=False, deterministic=True)

    def loadConfig(self, config_name):
        with open(config_name, 'r') as stream:
            self.cfg = yaml.safe_load(stream)

    def restore(self, model_name):
        self.player.restore(model_name)

    def getAction(self, state):
        self.obs['state'] = state
        return self.player.get_action(self.obs, is_deterministic=True).cpu().numpy()


class MuJoCoEnv:
    def __init__(self):
        self.createModel()
        self.initializeModel()
        self.setupPhysics()
        self.initForceAnchors()
        self.initializeLoggers()
        self.goal = np.zeros((2), dtype=np.float32)

        self.state = torch.zeros((1,10), dtype=torch.float32, device="cuda")

    def initializeModel(self):
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"top")

    def setupPhysics(self):
        self.model.opt.timestep = 0.02
        self.model.opt.gravity = [0,0,0]
        self.duration = 30

    def initializeLoggers(self):
        self.timevals = []
        self.angular_velocity = []
        self.linear_velocity = []
        self.position = []
        self.heading = []

    def createModel(self):
        sphere = """
        <mujoco model="tippe top">
          <option integrator="RK4"/>
        
          <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
             rgb2=".2 .3 .4" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
          </asset>
        
          <worldbody>
            <geom size="10.0 10.0 .01" type="plane" material="grid"/>
            <light pos="0 0 10.0"/>
            <camera name="closeup" pos="0 -3 2" xyaxes="1 0 0 0 1 2"/>
            <body name="top" pos="0 0 .4">
              <freejoint/>
              <geom name="ball" type="sphere" size=".31" mass="10.94"/>
            </body>
          </worldbody>
        
          <keyframe>
            <key name="idle" qpos="3 3 0.4 1 0 0 0" qvel="0 0 0 0 0 0" />
          </keyframe>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(sphere)

    def initForceAnchors(self):
        self.forces = np.array([[ 1, -1, 0],
                           [-1,  1, 0],
                           [ 1,  1, 0],
                           [-1, -1, 0],
                           [-1,  1, 0],
                           [ 1, -1, 0],
                           [-1, -1, 0],
                           [ 1,  1, 0]])
        
        self.positions = np.array([[ 1,  1, 0],
                              [ 1,  1, 0],
                              [-1,  1, 0],
                              [-1,  1, 0],
                              [-1, -1, 0],
                              [-1, -1, 0],
                              [ 1, -1, 0],
                              [ 1, -1, 0]]) * 0.2192


    def resetPosition(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def applyForces(self, action):
        self.data.qfrc_applied[...] = 0
        rmat = self.data.xmat[self.body_id].reshape(3,3)
        p = self.data.xpos[self.body_id]
        factor = max(np.sum(action), 1)
        for i in range(8):
          force = action[i] * (1./factor) * self.forces[i] * np.sqrt(0.5)
          if np.sum(np.abs(force)) > 0:
              force = np.matmul(rmat, force)
              p2 = np.matmul(rmat, self.positions[i]) + p
              mujoco.mj_applyFT(self.model, self.data, force, [0,0,0], p2, self.body_id, self.data.qfrc_applied)

    def updateLoggers(self):
        self.timevals.append(self.data.time)
        self.angular_velocity.append(self.data.qvel[3:6].copy())
        self.linear_velocity.append(self.data.qvel[0:3].copy())
        self.position.append(self.data.qpos[0:3].copy())

    def updateState(self):
        qpos = self.data.qpos.copy()
        siny_cosp = 2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5])
        cosy_cosp = 1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6])
        orient_z = np.arctan2(siny_cosp, cosy_cosp)
        dist_to_goal = self.goal - qpos[:2]
        angular_velocity = self.data.qvel[5].copy()
        linear_velocity = self.data.qvel[0:2].copy()
        self.state[0,0] = np.cos(orient_z)
        self.state[0,1] = np.sin(orient_z)
        self.state[0,2:4] = torch.tensor(linear_velocity, dtype=torch.float32, device="cuda")
        self.state[0,4] = angular_velocity
        self.state[0,5] = 0
        self.state[0,6:8] = torch.tensor(dist_to_goal, dtype=torch.float32, device="cuda")

    def runLoop(self, model, xy):
        self.resetPosition()
        self.data.qpos[:2] = xy
        mujoco.mj_step(self.model, self.data)
        while self.duration > self.data.time:
            self.updateState()
            action = model.getAction(self.state)
            self.applyForces(action)
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
                self.updateLoggers()

model = RLGamesModel()
model.loadConfig(config_name)
model.buildModel()
model.restore(model_name)

env = MuJoCoEnv()
env.runLoop(model, [4,0])

dpi = 120
width = 600
height = 800
figsize = (width / dpi, height / dpi)

fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

ax[0].plot(env.timevals, env.angular_velocity)
ax[0].set_title('angular velocity')
ax[0].set_ylabel('radians / second')

ax[1].plot(env.timevals, env.linear_velocity)
ax[1].set_xlabel('time (seconds)')
ax[1].set_ylabel('meters / second')
_ = ax[1].set_title('linear_velocity')
fig.savefig("test_velocities.png")

fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
ax[0].plot(env.timevals, env.position)
ax[0].set_xlabel('time (seconds)')
ax[0].set_ylabel('meters')
_ = ax[0].set_title('position')
ax[0].set_yscale('log')


ax[1].plot(np.array(env.position)[:,0], np.array(env.position)[:,1])
ax[1].set_xlabel('meters')
ax[1].set_ylabel('meters')
_ = ax[1].set_title('x y coordinates')
plt.tight_layout()
fig.savefig("test_positions.png")
