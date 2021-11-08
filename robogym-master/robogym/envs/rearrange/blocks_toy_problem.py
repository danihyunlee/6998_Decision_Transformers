import logging
import numpy as np
import attr

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)
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
for _ in range(10000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation.keys())
    obj_pos = observation['obj_pos']
    obj_rot = observation['obj_rot']
    goal_pos = observation['goal_obj_pos']
    goal_rot = observation['goal_obj_rot']
    l2_reward = 0
    for i in range(len(obj_pos)):
        l2_reward -= np.linalg.norm(np.array(obj_pos[i])-np.array(goal_pos[i]))
    print(l2_reward)

env.close()
print("done!")

    