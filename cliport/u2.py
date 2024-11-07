# set GPUimport os
import sys
import json

import numpy as np
from cliport import tasks
from cliport import agents
from cliport.utils import utils

import torch
import cv2
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
from cliport import tasks
from cliport.tasks import cameras
from cliport.utils import pybullet_utils
from cliport.utils import utils

root_dir = os.environ['CLIPORT_ROOT']
assets_root = os.path.join(root_dir, 'cliport/cliport/environments/assets/')

UR5_URDF_PATH = 'ur5/ur5.urdf'
UR5_WORKSPACE_URDF_PATH = 'ur5/workspace.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'

import pybullet as p
import pybullet
import tempfile
import time

# Initialize environment and task.
env = Environment(
    assets_root,
    disp=True,
    shared_memory=False,
    hz=480,
    record_cfg=False
)

task = tasks.names['place-red-in-green']()
env.set_task(task)
env.reset()
client = p.connect(pybullet.DIRECT, options="--msaa_samples=4")
p.setGravity(0, 0, -9.8)

file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
if file_io >= 0:
    p.executePluginCommand(
        file_io,
        textArgument=assets_root,
        intArgs=[p.AddFileIOAction],
        physicsClientId=client)

plane = p.loadURDF(os.path.join(assets_root, PLANE_URDF_PATH),
                                 [0, 0, -0.001])

shade = np.random.rand() + 0.5
color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
p.changeVisualShape(plane, -1, rgbaColor=color)

table = pybullet_utils.load_urdf(
            p, os.path.join(assets_root, UR5_WORKSPACE_URDF_PATH), [0.4, 0, 0])
p.changeVisualShape(table, -1, rgbaColor=[1, 0, 0, 1])

agent_cam = cameras.RealSenseD415.CONFIG
config = agent_cam[0]

lookdir = np.float32([0, 0, 1]).reshape(3, 1)
updir = np.float32([0, -1, 0]).reshape(3, 1)
rotation = p.getMatrixFromQuaternion(config['rotation'])
rotm = np.float32(rotation).reshape(3, 3)
lookdir = (rotm @ lookdir).reshape(-1)
updir = (rotm @ updir).reshape(-1)
lookat = config['position'] + lookdir
focal_len = config['intrinsics'][0]
znear, zfar = config['zrange']
viewm = p.computeViewMatrix(config['position'], lookat, updir)
fovh = (config['image_size'][1] / 2) / focal_len
fovh = 180 * np.arctan(fovh) * 2 / np.pi
# Notes: 1) FOV is vertical FOV 2) aspect must be float
aspect_ratio = config['image_size'][1] / config['image_size'][1]
projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

_, _, color, depth, segm = p.getCameraImage(
    width=config['image_size'][1],
    height=config['image_size'][0],
    viewMatrix=viewm,
    projectionMatrix=projm,
    shadow=False,
    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    renderer=p.ER_BULLET_HARDWARE_OPENGL)

plt.imsave('color.png', color)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
p.disconnect()


# plane = p.loadURDF(os.path.join(assets_root, PLANE_URDF_PATH),
#                                  [0, 0, -0.001])

# # 테이블
# table = pybullet_utils.load_urdf(
#             p, os.path.join(assets_root, UR5_WORKSPACE_URDF_PATH), [0.4, 0, 0])

# visual_shape_id = p.createVisualShape(
#     shapeType=p.GEOM_BOX,
#     rgbaColor=[1, 0, 0, 1],  # Red color (R, G, B, Alpha)
#     halfExtents=[0.5, 0.5, 0.05]
# )
# collision_shape_id = p.createCollisionShape(
#     shapeType=p.GEOM_BOX,
#     halfExtents=[0.5, 0.5, 0.05]
# )
# p.createMultiBody(
#     baseMass=0,
#     baseCollisionShapeIndex=collision_shape_id,
#     baseVisualShapeIndex=visual_shape_id,
#     basePosition=[0, 0, 0.05]
# )

# ur5 = pybullet_utils.load_urdf(p, os.path.join(assets_root, UR5_URDF_PATH))
# n_joints = p.getNumJoints(ur5)
# joints = [p.getJointInfo(ur5, i) for i in range(n_joints)]
# joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.configureDebugVisualizer(rgbBackground=[0, 0, 0])

# import time
# time.sleep(50)

# p.disconnect()