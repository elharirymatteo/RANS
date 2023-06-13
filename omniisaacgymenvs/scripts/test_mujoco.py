import mujoco
#@title Import packages for plotting and creating graphics
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt

import pandas as pd

actions = pd.read_csv("/home/antoine/Downloads/actions.csv")
actions = actions.to_numpy()[50:,1:]
actions[:,:] = 0
actions[:,1] = 1
actions[:,3] = 1
actions[:,5] = 1
actions[:,7] = 1
#print(actions.shape)
# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

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
    <key name="idle" qpos="10 10 0.4 1 0 0 0" qvel="0 0 0 0 0 0" />
  </keyframe>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(sphere)
renderer = mujoco.Renderer(model)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera="closeup")
media.show_image(renderer.render())
media.write_image('/tmp/sphere.png', renderer.render())

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"top")

model.opt.timestep = 0.002
model.opt.gravity = [0,0,0]

print('initial_positions', data.qpos)
print('initial_velocities', data.qvel)

timevals = []
angular_velocity = []
linear_velocity = []
position = []
duration = 1
print(model.opt.timestep)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"top")

forces = np.array([[ 1, -1, 0],
                  [-1,  1, 0],
                  [ 1,  1, 0],
                  [-1, -1, 0],
                  [-1,  1, 0],
                  [ 1, -1, 0],
                  [-1, -1, 0],
                  [ 1,  1, 0]])

positions = np.array([[ 1,  1, 0],
                      [ 1,  1, 0],
                      [-1,  1, 0],
                      [-1,  1, 0],
                      [-1, -1, 0],
                      [-1, -1, 0],
                      [ 1, -1, 0],
                      [ 1, -1, 0]]) * 0.2192

positions[:,2]

# Simulate and save data
mujoco.mj_resetDataKeyframe(model, data, 0)
timevals
for act in actions[:2]:
  for _ in range(100):
    mujoco.mj_step(model, data)
  data.qfrc_applied[...] = 0
  timevals.append(data.time)
  rmat = data.xmat[body_id].reshape(3,3)
  p = data.xpos[body_id]

  # Rescale thrust factor
  factor = max(np.sum(act), 1)
  for i in range(8):
    force = act[i] * (1./factor) * forces[i] * np.sqrt(0.5)
    if np.sum(np.abs(force)) > 0:
        force = np.matmul(rmat, force)
        p2 = np.matmul(rmat, positions[i]) + p
        mujoco.mj_applyFT(model, data, force, [0,0,0], p2, body_id, data.qfrc_applied)

  angular_velocity.append(data.qvel[3:6].copy())
  linear_velocity.append(data.qvel[0:3].copy())
  position.append(data.qpos[0:3].copy())
print(angular_velocity[-1])

dpi = 120
width = 600
height = 800
figsize = (width / dpi, height / dpi)
_, ax = plt.subplots(4, 1, figsize=figsize, dpi=dpi)

ax[0].plot(timevals, angular_velocity)
ax[0].set_title('angular velocity')
ax[0].set_ylabel('radians / second')

ax[1].plot(timevals, linear_velocity)
ax[1].set_xlabel('time (seconds)')
ax[1].set_ylabel('meters / second')
_ = ax[1].set_title('linear_velocity')

ax[2].plot(timevals, position)
ax[2].set_xlabel('time (seconds)')
ax[2].set_ylabel('meters')
_ = ax[2].set_title('position')

ax[3].plot(np.array(position)[:,0], np.array(position)[:,1])
ax[3].set_xlabel('meters')
ax[3].set_ylabel('meters')
_ = ax[3].set_title('x y coordinates')
plt.show()
