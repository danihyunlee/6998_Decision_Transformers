import logging
import numpy as np
import attr

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)

from time import sleep

from robogym.envs.rearrange.goals.object_stack_goal import ObjectStackGoal
from robogym.envs.rearrange.simulation.base import RearrangeSimParameters
from robogym.envs.rearrange.simulation.blocks import (
    BlockRearrangeSim,
    BlockRearrangeSimParameters,
)

from robogym.robot.composite.controllers.ur_gripper_arm import (
    Direction,
    URGripperArmController,
)
from robogym.robot_env import build_nested_attr

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class BlockStackEnvConstants(RearrangeEnvConstants):
    # whether block stacked in the fixed order or random order.
    fixed_order: bool = False


@attr.s(auto_attribs=True)
class BlockStackEnvParameters(RearrangeEnvParameters):
    simulation_params: RearrangeSimParameters = build_nested_attr(
        BlockRearrangeSimParameters, default=dict(num_objects=1)
    )


class BlockStackEnv(
    RearrangeEnv[BlockStackEnvParameters, BlockStackEnvConstants, BlockRearrangeSim]
):
    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        return ObjectStackGoal(
            mujoco_simulation, constants.goal_args, constants.fixed_order
        )

# Our implementation starts here

make_env = BlockStackEnv.build
env = make_env()
controller = URGripperArmController(env.unwrapped)
robot = env.robot # environment's robot with 6DOF
#print(env.unwrapped.mujoco_simulation.)
obs = env.reset()
#print(env.action_space)

"""
All potential information we can get from thh observation

['obj_pos', 'obj_rel_pos', 'obj_vel_pos', 'obj_rot', 'obj_vel_rot', 'robot_joint_pos', \
'gripper_pos', 'gripper_velp', 'gripper_controls', 'gripper_qpos', 'gripper_vel', 'qpos', \
'qpos_goal', 'goal_obj_pos', 'goal_obj_rot', 'is_goal_achieved', 'rel_goal_obj_pos', \
'rel_goal_obj_rot', 'obj_gripper_contact', 'obj_bbox_size', 'obj_colors', 'safety_stop', \
'tcp_force', 'tcp_torque', 'action_ema']
"""

env.reset()
old_reward = 0
for _ in range(20):
    env.render()

    # Image Extraction
    a1 = env.render(mode="rgb_array", width=300,height=300) # Default renderer from robogym/robot_env.py class
    a2 = env.sim.render(camera_name='vision_cam_top', width=300, height=300, depth=True) # Mujoco specific renderer
    # Cameras available include: 'vision_cam_wrist', 'vision_cam_top', 'vision_cam_front', 'phys_checks_cam'
    # TODO: How can we remove the goal shadow from the image? How can we add new types of cameras/change it?
    # Option we can use depth image data to easily bypass the goal shadow (goal shadow has no depth)
    # 
    
    x = env.observe()
    #print(x.keys())
    #print("\n\n")
    action = env.action_space.sample()
    # TODO: What is the meaning of this action space and why is it a discrete 11?
    # How can we make this non-discrete and encode instead other inputs?
    #print(env.action_space)
    observation, reward, done, info = env.step(action)
    print(reward)

    obj_pos = observation['obj_pos']
    print(obj_pos)
    obj_rot = observation['obj_rot']
    goal_pos = observation['goal_obj_pos']
    goal_rot = observation['goal_obj_rot']
    l2_reward = 0
    # TODO: Encode the reward function as an environment function and return from the iteration.
    for i in range(len(obj_pos)):
        l2_reward -= np.linalg.norm(np.array(obj_pos[i])-np.array(goal_pos[i]))
        new_reward = l2_reward
        _reward = new_reward - old_reward
        old_reward = new_reward

    print(_reward)

env.close()

import matplotlib.pyplot as plt

# Visualization of mujoco camera image
print(np.shape(a2[1]))
plt.imshow(a1)
plt.show()
plt.imshow(a2[1])
plt.show()
print("done!")

