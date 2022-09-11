import random
import numpy as np
import os
import pickle
from queue import PriorityQueue
import torch


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=1):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print("Saving buffer to {}".format(save_path))

        with open(save_path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print("Loading buffer from {}".format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


def sample_model_and_store(agent, memory, memory_model, batch_size):
    (
        state_batch,
        action_batch1,
        reward_batch1,
        next_state_batch1,
        mask_batch1,
    ) = memory.sample(batch_size=batch_size)

    batch = [state_batch, action_batch1, reward_batch1, next_state_batch1, mask_batch1]

    state_batch = torch.FloatTensor(batch[0]).to(agent.device)
    next_state_batch = torch.FloatTensor(batch[3]).to(agent.device)
    action_batch = torch.FloatTensor(batch[1]).to(agent.device)
    reward_batch = torch.FloatTensor(batch[2]).to(agent.device).unsqueeze(1)
    mask_batch = torch.FloatTensor(batch[4]).to(agent.device).unsqueeze(1)

    with torch.no_grad():
        action_batch, __, __ = agent.policy.sample(state_batch)
        action_batch = action_batch.detach().cpu().numpy()[agent.active_ensemble]
        action_batch = torch.FloatTensor(action_batch).to(agent.device)
        next_state_batch, reward_batch, mask_batch = agent.model.sample(
            state_batch, action_batch
        )
    for i in range(batch_size):
        memory_model.push(
            state_batch[i, :].cpu(),
            action_batch[i, :].cpu(),
            reward_batch[i, :].cpu(),
            next_state_batch[i, :].cpu(),
            mask_batch[i, :].cpu(),
        )  # Append transition to memory

    return


def join_modelsim_data(agent, memory, memory_model, batch_size, ratio):

    if ratio == 0.0:
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            mask_batch,
        ) = memory.sample(batch_size=int(batch_size))

        batch = [state_batch, action_batch, reward_batch, next_state_batch, mask_batch]

        return batch

    (
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        mask_batch,
    ) = memory_model.sample(batch_size=int(ratio * batch_size))

    (
        state_batch1,
        action_batch1,
        reward_batch1,
        next_state_batch1,
        mask_batch1,
    ) = memory.sample(batch_size=batch_size - int(ratio * batch_size))

    batch = [
        state_batch1,
        action_batch1,
        reward_batch1[..., np.newaxis],
        next_state_batch1,
        mask_batch1[..., np.newaxis],
    ]

    state_batch = np.concatenate((state_batch, batch[0]), axis=0)
    action_batch = np.concatenate((action_batch, batch[1]), axis=0)

    reward_batch = np.concatenate((reward_batch, batch[2]), axis=0)
    next_state_batch = np.concatenate((next_state_batch, batch[3]), axis=0)
    mask_batch = np.concatenate((mask_batch, batch[4]), axis=0)

    batch = [
        state_batch,
        action_batch,
        reward_batch.squeeze(),
        next_state_batch,
        mask_batch.squeeze(),
    ]

    return batch
