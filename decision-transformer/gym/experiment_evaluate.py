import gym
import numpy as np
import torch
import wandb
import os
import d4rl
import torch
import torch.nn as nn

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel

root = '/proj/vondrick2/james/robotics/'
# root = './'

"""sample script: python experiment_evaluate.py --env kitchen-complete --model_savepath ../../ --model_name dt_kitchen-complete_rgbd_2.pt --device cpu --train_with_rgb true --train_with_depth true --num_eval_episodes 10"""


device = 'cpu'
def fgsm_attack(model, loss, images, labels, eps) :
    
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images

class RGB_Encoder(nn.Module):    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(27 * 27 * 32, 128),
            # nn.Linear(23328,128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class RGB_Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 27 * 27 * 32),
            # nn.Linear(128,23328),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 27, 27))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class RGB_CAE(nn.Module):
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.encoder = RGB_Encoder(encoded_space_dim, fc2_input_dim)
        self.decoder = RGB_Decoder(encoded_space_dim, fc2_input_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        # print("LATENT SPACE")
        # print(x)
        x = self.decoder(x)
        return x


class Depth_Encoder(nn.Module):
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(27 * 27 * 32, 128),
            # nn.Linear(23328,128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Depth_Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 27 * 27 * 32),
            # nn.Linear(128,23328),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 27, 27))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class Depth_CAE(nn.Module):
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.encoder = Depth_Encoder(encoded_space_dim, fc2_input_dim)
        self.decoder = Depth_Decoder(encoded_space_dim, fc2_input_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        # print("LATENT SPACE")
        # print(x)
        x = self.decoder(x)
        return x


def evaluate_episode_bc_rgbd(
        state_init,
        variant,
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        visualize = False,
        img_encoder=None,
        depth_encoder=None
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()


    #init_qpos = state[:30].copy()
    init_qpos = state_init[:30].copy()
    init_qpos = init_qpos + np.random.normal(0, 0.001, size=init_qpos.shape)

    # prepare env

    env.sim.model.key_qpos[:] = init_qpos
    env.sim.forward()

    state[:30] = init_qpos
    state = state.reshape(1,60)

    if variant['train_with_objpose'] == False:
        state = state[:,:9]

    if variant['train_with_rgb']:
        rgb_input = env.render(mode='rgb_array', depth=False)
        """ Add in adversarial attacks for rgb_input here"""
        
        rgb_input = rgb_input.reshape(1,rgb_input.shape[0],rgb_input.shape[1],3)
        rgb_input = torch.tensor(rgb_input.transpose(0,3,1,2).copy())
        encoded_input = img_encoder(rgb_input.float()).detach().numpy()
        encoded_input = encoded_input.reshape(1,32)
        state = np.concatenate((state,encoded_input),axis = 1)

    if variant['train_with_depth']:
        depth_input = env.render(mode='rgb_array', depth=True)
        depth_input = depth_input.reshape(1,depth_input.shape[0],depth_input.shape[1],1)
        depth_input = torch.tensor(depth_input.transpose(0,3,1,2).copy())
        encoded_input = depth_encoder(depth_input.float()).detach().numpy()
        encoded_input = encoded_input.reshape(1,32)
        state = np.concatenate((state,encoded_input),axis = 1)


    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0

    total_reward = 0
    total_return = 0
    for t in range(max_ep_len):
        # visualize environemnt
        if (visualize == True):
            env.render(mode='human')
            
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()
        state, reward, done, info = env.step(action)
        state = state.reshape(1,60)
        # qp 9, obj_qp 21, goal 30
        
        rgb_input = None
        depth_input = None
        if variant['train_with_objpose'] == False:
            state = state[:,:9]

        if variant['train_with_rgb']:
            rgb_input = env.render(mode='rgb_array', depth=False)

            """ Add in adversarial attacks for rgb_input here"""

            rgb_input = rgb_input.reshape(1,rgb_input.shape[0],rgb_input.shape[1],3)
            rgb_input = torch.tensor(rgb_input.transpose(0,3,1,2).copy())
            encoded_input = img_encoder(rgb_input.float()).detach().numpy()
            encoded_input = encoded_input.reshape(1,32)
            state = np.concatenate((state,encoded_input),axis = 1)

        if variant['train_with_depth']:
            depth_input = env.render(mode='rgb_array', depth=True)
            depth_input = depth_input.reshape(1,depth_input.shape[0],depth_input.shape[1],1)
            depth_input = torch.tensor(depth_input.transpose(0,3,1,2).copy())
            encoded_input = depth_encoder(depth_input.float()).detach().numpy()
            encoded_input = encoded_input.reshape(1,32)
            state = np.concatenate((state,encoded_input),axis = 1)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)

        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        if reward == 1:
            total_reward += 1
        total_return += total_reward

        episode_length += 1

        if done:
            break
    return episode_return, episode_length, total_return


def evaluate_episode_dt_rgbd(
        state_init,
        variant,
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        visualize = False,
        img_encoder=None,
        depth_encoder=None
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()


    #init_qpos = state[:30].copy()
    init_qpos = state_init[:30].copy()
    #init_qpos = init_qpos + np.random.normal(0, 0.001, size=init_qpos.shape)

    # prepare env

    env.sim.model.key_qpos[:] = init_qpos
    env.sim.forward()

    state[:30] = init_qpos
    state = state.reshape(1,60)

    if variant['train_with_objpose'] == False:
        state = state[:,:9]

    if variant['train_with_rgb']:
        rgb_input = env.render(mode='rgb_array', depth=False)
        """ Add in adversarial attacks for rgb_input here"""
        
        rgb_input = rgb_input.reshape(1,rgb_input.shape[0],rgb_input.shape[1],3)
        rgb_input = torch.tensor(rgb_input.transpose(0,3,1,2).copy())
        encoded_input = img_encoder(rgb_input.float()).detach().numpy()
        encoded_input = encoded_input.reshape(1,32)
        state = np.concatenate((state,encoded_input),axis = 1)

    if variant['train_with_depth']:
        depth_input = env.render(mode='rgb_array', depth=True)
        depth_input = depth_input.reshape(1,depth_input.shape[0],depth_input.shape[1],1)
        depth_input = torch.tensor(depth_input.transpose(0,3,1,2).copy())
        encoded_input = depth_encoder(depth_input.float()).detach().numpy()
        encoded_input = encoded_input.reshape(1,32)
        state = np.concatenate((state,encoded_input),axis = 1)

    #if mode == 'noise':
    # state = state + np.random.normal(0, 0.001, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)

    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0

    total_reward = 0
    total_return = 0
    for t in range(max_ep_len):
        # visualize environemnt
        if (visualize == True):
            # """
            # depth_img = env.render(mode='rgb_array', depth=True)
            # plt.imshow(depth_img)
            # plt.savefig('./'+str(t)+'_depth.png')
            # rgb_img = env.render(mode='rgb_array', depth=False)
            # plt.imshow(rgb_img)
            # plt.savefig('./'+str(t)+'_rgb.png')
            # """
            env.render(mode='human')
            
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()
        state, reward, done, info = env.step(action)
        state = state.reshape(1,60)
        # qp 9, obj_qp 21, goal 30
        
        rgb_input = None
        depth_input = None
        if variant['train_with_objpose'] == False:
            state = state[:,:9]

        if variant['train_with_rgb']:
            rgb_input = env.render(mode='rgb_array', depth=False)
            rgb_input = rgb_input.reshape(1,rgb_input.shape[0],rgb_input.shape[1],3)
            rgb_input = torch.tensor(rgb_input.transpose(0,3,1,2).copy()).float()

            if variant['white_noise_attack_rgb']:
                attack_rgb_model = RGB_CAE(encoded_space_dim=32,fc2_input_dim=128)

                all_state_dict = {}
                encoder_weights = torch.load("../../img_depth_encoder/encoder_img.pth", map_location=torch.device('cpu'))
                for k in encoder_weights:
                    all_state_dict['encoder.' + k] = encoder_weights[k]
                    
                decoder_weights = torch.load("../../img_depth_encoder/decoder_img.pth", map_location=torch.device('cpu'))
                for k in decoder_weights:
                    all_state_dict['decoder.' + k] = decoder_weights[k]

                attack_rgb_model.load_state_dict(all_state_dict)

            """ Add in adversarial attacks for rgb_input here"""

            if variant['white_noise_attack_rgb']:
                loss = torch.nn.MSELoss()
                adv_input = fgsm_attack(attack_rgb_model, loss, rgb_input, rgb_input, eps=0.03).to(device)
                encoded_input = img_encoder(adv_input).detach().numpy()
            else:
                encoded_input = img_encoder(rgb_input).detach().numpy()

            encoded_input = encoded_input.reshape(1,32)
            state = np.concatenate((state,encoded_input),axis = 1)

        if variant['train_with_depth']:
            depth_input = env.render(mode='rgb_array', depth=True)
            depth_input = depth_input.reshape(1,depth_input.shape[0],depth_input.shape[1],1)
            depth_input = torch.tensor(depth_input.transpose(0,3,1,2).copy()).float()

            if variant['white_noise_attack_depth']:
                attack_depth_model = Depth_CAE(encoded_space_dim=32,fc2_input_dim=128)

                all_state_dict = {}
                encoder_weights = torch.load("../../img_depth_encoder/encoder_depth.pth", map_location=torch.device('cpu'))
                for k in encoder_weights:
                    all_state_dict['encoder.' + k] = encoder_weights[k]
                    
                decoder_weights = torch.load("../../img_depth_encoder/decoder_depth.pth", map_location=torch.device('cpu'))
                for k in decoder_weights:
                    all_state_dict['decoder.' + k] = decoder_weights[k]

                attack_depth_model.load_state_dict(all_state_dict)


            if variant['white_noise_attack_depth']:
                loss = torch.nn.MSELoss()
                adv_input = fgsm_attack(attack_depth_model, loss, depth_input, depth_input, eps=0.03).to(device)
                encoded_input = img_encoder(adv_input).detach().numpy()
            else:
                encoded_input = depth_encoder(depth_input).detach().numpy()
                
            encoded_input = encoded_input.reshape(1,32)
            state = np.concatenate((state,encoded_input),axis = 1)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)

        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        if reward == 1:
            total_reward += 1
        total_return += total_reward

        episode_length += 1

        if done:
            break
    return episode_return, episode_length, total_return

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment_evaluate(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    elif env_name == 'kitchen-complete':
        env = gym.make('kitchen-complete-v0')
        max_ep_len = 206
        env_targets = [4] # placeholder for now, not relevant unless we work with multitask learning
        scale = 10. # placeholder for now
    elif env_name == 'kitchen-partial':
        env = gym.make('kitchen-partial-v0')
        max_ep_len = 527
        env_targets = [100, 50]
        scale = 10.
    elif env_name == 'kitchen-mixed':
        env = gym.make('kitchen-mixed-v0')
        max_ep_len = 527
        env_targets = [100, 50]
        scale = 10.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    original_state_dim = env.observation_space.shape[0]
    img_encoder = None
    depth_encoder = None

    if variant['train_with_objpose'] == False:
        state_dim -= 51

    if variant['train_with_rgb']:
        state_dim += 32   # img 32*1 dim
        img_encoder = RGB_Encoder(encoded_space_dim=32,fc2_input_dim=128)
        img_encoder.load_state_dict(torch.load("../../img_depth_encoder/encoder_img.pth",map_location=variant['device']))
        img_encoder.to(device=device)


    if variant['train_with_depth']:
        state_dim += 32   # depth 32*1 dim
        depth_encoder = Depth_Encoder(encoded_space_dim=32,fc2_input_dim=128)
        depth_encoder.load_state_dict(torch.load("../../img_depth_encoder/encoder_depth.pth",map_location=variant['device']))
        depth_encoder.to(device=device)


    act_dim = env.action_space.shape[0]

    # load dataset

    print(env_name)

    if(env_name == 'kitchen-complete'):
        dataset_path = os.path.join(root, f'data/kitchen-complete-v0.pkl')
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif(env_name == 'kitchen-mixed'):
        dataset_path = os.path.join(root, f'data/kitchen-mixed-v0.pkl')
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif(env_name == 'kitchen-partial'):
        dataset_path = os.path.join(root, f'data/kitchen-partial-v0.pkl')
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    else:
        dataset_path = os.path.join(root, f'data/{env_name}-{dataset}-v0.pkl')
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    if variant['train_with_objpose'] == False:
        state_mean = state_mean[:9]
        state_std = state_std[:9]
    if (env_name == 'kitchen-complete'):
        num_padding = 32*((variant['train_with_rgb']==1)+(variant['train_with_depth']==1))
        state_mean = np.pad(state_mean, (0, num_padding), 'constant', constant_values=(0, 0))
        state_std = np.pad(state_std, (0, num_padding), 'constant', constant_values=(1,1))
    
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    def eval_episodes(variant, target_rew, states, img_encoder=None, depth_encoder=None):
        returns, lengths, tot_returns = [], [], []
        for _ in range(num_eval_episodes):
            state_init = random.choice(states).copy()
            with torch.no_grad():
                # Added in code for evaluation in kitchen for DT
                if env_name == 'kitchen-complete' and model_type == 'dt':
                    ret, length, tot_ret = evaluate_episode_dt_rgbd(
                        state_init,
                        variant,
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        img_encoder=img_encoder,
                        depth_encoder=depth_encoder,
                        visualize=variant['visualize']
                    )
                else:
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        if env_name == 'kitchen-complete' and model_type == 'bc':
                            ret, length, tot_ret = evaluate_episode_bc_rgbd(
                                state_init,
                                variant,
                                env,
                                state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                scale=scale,
                                target_return=target_rew/scale,
                                mode=mode,
                                state_mean=state_mean,
                                state_std=state_std,
                                device=device,
                                img_encoder=img_encoder,
                                depth_encoder=depth_encoder,
                                visualize=variant['visualize']
                            )
                        else:    
                            ret, length = evaluate_episode(
                                env,
                                state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                target_return=target_rew/scale,
                                mode=mode,
                                state_mean=state_mean,
                                state_std=state_std,
                                device=device,
                            )
                returns.append(ret)
                lengths.append(length)
                tot_returns.append(tot_ret)
                print('episode return', ret, 'episode length', length, 'total return', tot_ret)
        return{
            f'target_{target_rew}_return_mean': np.mean(returns),
            f'target_{target_rew}_return_std': np.std(returns),
            f'target_{target_rew}_length_mean': np.mean(lengths),
            f'target_{target_rew}_length_std': np.std(lengths),
            f'target_{target_rew}_total_return_mean': np.mean(tot_returns),
            f'target_{target_rew}_total_return_std': np.std(tot_returns),
        }

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    # Loading saved model
    model.load_state_dict(torch.load(variant['model_savepath']+variant['model_name'],map_location=variant['device']))
    model.eval()
    model.to(device=device)

    # Evaluate
    if (variant['env']=='kitchen-complete'):
        print(eval_episodes(variant, env_targets[0], states, img_encoder=img_encoder, depth_encoder=depth_encoder))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--model_savepath', type=str, default=os.path.join('.','models'))
    parser.add_argument('--model_name', type=str, default='')
    
    """ Added Parameters """
    parser.add_argument('--train_with_objpose', action='store_true')
    parser.add_argument('--train_with_depth', action='store_true')
    parser.add_argument('--train_with_rgb', action='store_true')
    parser.add_argument('--white_noise_attack_rgb', action='store_true')
    parser.add_argument('--white_noise_attack_depth', action='store_true')


    args = parser.parse_args()

    

    experiment_evaluate('gym-experiment', variant=vars(args))
