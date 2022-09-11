import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayMemory, sample_model_and_store, join_modelsim_data
from agent.ensembled_vaml import EnsembledVAML
import contextlib
import collections
from tqdm import tqdm
import hydra
import os
import utils
from logger import Logger

class Workspace(object):
    def __init__(self, cfg):
        super().__init__()
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = gym.make(cfg.env)
        self.env.seed(cfg.seed)
        self.env.action_space.seed(cfg.seed)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.memory = ReplayMemory(self.env.observation_space.shape,
                                    self.env.action_space.shape,
                                    int(cfg.replay_buffer_capacity),
                                    self.device)
        self.memory_model = ReplayMemory(self.env.observation_space.shape,
                                    self.env.action_space.shape,
                                    int(cfg.replay_buffer_capacity),
                                    self.device)
        
        self.total_numsteps = 0
        self.updates = 0
        self.episodes = 0

    def evaluate(self):
        avg_episode_reward = 0.0
        for episode in range(self.cfg.num_eval_episodes):
            state = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            while not done:
                with torch.no_grad():
                    action = self.agent.select_action(state, evaluate=True)
                next_step, reward, done, _ = self.env.step(action)
                episode_reward += reward

            avg_episode_reward += episode_reward
        avg_episode_reward /= self.cfg.num_eval_episodes

        self.logger.log('avg_reward/test', avg_episode_reward,
                        self.episodes)
        print("----------------------------------------")
        print(
            "Test Episodes: {}, Avg. Reward: {}".format(
            self.episodes, round(avg_episode_reward, 2)
            )
        )
        print("----------------------------------------")
        self.agent.save_checkpoint(self.cfg.env_name , suffix = "{}_{}_{}".format(self.cfg.simname, self.cfg.seed, self.cfg.ratio))


    def ep_train(self):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = self.env.reset()
        self.agent.reset()
        while not done:
            if len(self.memory) > self.cfg.batch_size:
                for i in range(self.cfg.updates_per_step):   
                    if self.total_numsteps < self.cfg.model_use_thres:
                        sample_model_and_store(self.agent, self.memory, self.memory_model, self.cfg.batch_size)
                        batch = join_modelsim_data(self.agent, self.memory, self.memory_model, self.cfg.batch_size, ratio = 0.0)
                        stats = self.agent.update_params_sim(batch, self.updates)
                    else:
                        sample_model_and_store(self.agent, self.memory, self.memory_model, self.cfg.batch_size)
                        batch = join_modelsim_data(self.agent, self.memory, self.memory_model, self.cfg.batch_size, ratio = self.cfg.ratio)
                        stats = self.agent.update_params_sim(batch, self.updates)


                    self.logger.log("loss/critic_1", stats[0], self.updates)
                    self.logger.log("loss/critic_2", stats[1], self.updates)
                    self.logger.log("loss/policy", stats[2], self.updates)
                    self.logger.log("loss/entropy_loss", stats[3], self.updates)
                    self.logger.log("entropy_temprature/alpha", stats[4], self.updates)

                    for _ in range(self.cfg.model_updates_per_step):
                        model_stats = self.cfgagent.update_model(self.memory, self.cfg.batch_size)
                        self.logger.log("loss/model_loss", model_stats[0], self.updates)
                        self.logger.log("loss/reward_loss", model_stats[1], self.updates)
                        self.logger.log("loss/mask_loss", model_stats[2], self.updates)
                        self.logger.log("loss/mse_loss", model_stats[3], self.updates)

                self.updates += 1

            if self.cfg.start_steps > self.total_numsteps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(state)
            
            next_state, reward, done, _ = self.env.step(action)  # Step
            episode_steps += 1
            self.total_numsteps += 1
            episode_reward += reward
            mask = 1 if episode_steps == self.env._max_episode_steps else float(not done)

            self.memory.push(
                state, action, reward, next_state, mask
            )  # Append transition to memory

            state = next_state
        self.logger.log("reward/train", episode_reward, self.episode)
        print(
            "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
                self.episode, self.total_numsteps, episode_steps, round(episode_reward, 2)
            )
        )
        return


    def run(self):
        for i_episode in itertools.count(1):
            self.episodes = i_episode
            self.ep_train()
            if self.episodes %10 == 0:
                self.evaluate()
            if self.total_numsteps > self.cfg.num_steps:
                break
        return

    

    



        
        

