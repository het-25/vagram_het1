import math
import os
import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn


def make_env(cfg):
    """Helper function to create dm_control or gym environment"""

    if "gym___" in cfg.env:
        env = gym.make(cfg.env.split("___")[1])
    else:
        import mbrl.third_party.dmc2gym as dmc2gym

        if cfg.env == "ball_in_cup_catch":
            domain_name = "ball_in_cup"
            task_name = "catch"
        else:
            domain_name = cfg.env.split("_")[0]
            task_name = "_".join(cfg.env.split("_")[1:])

        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=cfg.seed,
            visualize_reward=True,
        )
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weights_init_)

    def forward(self, x):
        return self.trunk(x)

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, normalize=False):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        if normalize:
            mods.append(NormLayer())
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

class NormLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, dim=-1)

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def weights_init_(m):
    if isinstance(m, nn.Module):
        for k in m.modules():
            if isinstance(k, EnsembleLinearLayer) or isinstance(k, nn.Linear):
                torch.nn.init.orthogonal_(k.weight, 1)
                torch.nn.init.constant_(k.bias, 0)

class EnsembleLinearLayer(nn.Module):
    """
    Efficient linear layer for ensemble models.
    Taken from https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/models/util.py
    """

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=0)
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            self.use_bias = True
            torch.nn.init.zeros_(self.bias)
        else:
            self.use_bias = False

    def forward(self, x):
        if x.dim() == 2:
            # print("dim 2")
            # print(x.shape)
            # print(self.weight.shape)
            xw = x.matmul(self.weight)
            # print(xw.shape)
        else:
            # print("dim 3")
            # print(x.shape)
            # print(self.weight.shape)
            xw = torch.einsum("ebd,edm->ebm", x, self.weight)
            # print(xw.shape)
        if self.use_bias:
            return xw + self.bias
        else:
            return xw

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

