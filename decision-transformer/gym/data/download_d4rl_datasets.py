import gym
import numpy as np

import collections
import pickle
import os
import d4rl
import matplotlib.pyplot as plt

root = '/proj/vondrick2/james/robotics/data'
datasets = []
"""
for env_name in ['halfcheetah', 'hopper', 'walker2d']:
    for dataset_type in ['medium', 'medium-replay', 'expert']:
        name = f'{env_name}-{dataset_type}-v2'
        env = gym.make(name)
        dataset = env.get_dataset()

        N = dataset['rewards'].shape[0]
        data_ = collections.defaultdict(list)

        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True
        episode_step = 0
        paths = []
        for i in range(N):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000-1)
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
                print(episode_data['observations'][0])
            episode_step += 1

        returns = np.array([np.sum(p['rewards']) for p in paths])
        num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        print(f'Number of samples collected: {num_samples}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

        with open(os.path.join(root,f'{name}.pkl'), 'wb') as f:
            pickle.dump(paths, f)
"""


""" Added code for loading FrankaKitchen environment """
env_name = 'kitchen'
for dataset_type in ['complete', 'partial', 'mixed']:
    name = f'{env_name}-{dataset_type}-v0'
    env = gym.make(name)
    dataset = d4rl.qlearning_dataset(env);
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    print("Avaliable keys to dataset", dataset.keys())
    for i in range(N):
        
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000-1)
        for k in ['observations','next_observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(dataset[k][i])
        qpos_initial = data_['observations'][0][:30].copy()
        qvel_initial = np.array(data_['observations'][0][30:59].copy())

        all_keys = []
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            """ adding depth and rgb inputs to dataset """
            rgb_output = []
            depth_output = []
            env.reset()
            env.sim.data.qpos[:] = qpos_initial
            env.sim.data.qvel[:] = qvel_initial

            for i in range(episode_data['actions'].shape[0]):
                """ setting environment to timestep state and retrieving camera inputs """
                env.sim.data.qpos[:] = data_['observations'][i][:30].copy()
                env.sim.forward()
                rgb_output.append(env.render(mode='rgb_array', depth=False))
                depth_output.append(env.render(mode='rgb_array', depth=True))

            episode_data['rgb'] = rgb_output
            episode_data['depth'] = depth_output
            all_keys = episode_data.keys()
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Types of data collected: {all_keys}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    with open(os.path.join(root, f'{name}.pkl'), 'wb') as f:
        pickle.dump(paths, f)
