import numpy as np
import random
from collections import namedtuple, deque

from model import ModuleQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentObject():
    def __init__(self, data_state_size, data_action_size, data_seed):
        self.data_state_size = data_state_size
        self.data_action_size = data_action_size
        self.data_seed = random.seed(data_seed)
        self.data_qnetwork_local = ModuleQNetwork(data_state_size, data_action_size, data_seed).to(device)
        self.data_qnetwork_target = ModuleQNetwork(data_state_size, data_action_size, data_seed).to(device)
        self.data_optimizer = optim.Adam(self.data_qnetwork_local.parameters(), lr=LR)
        self.data_memory = ReplayBufferObject(data_action_size, BUFFER_SIZE, BATCH_SIZE, data_seed)
        self.data_t_step = 0

    def learn(self, data_experiences, data_gamma):
        data_states, data_actions, data_rewards, data_next_states, data_dones = data_experiences
        data_Q_targets_next = self.data_qnetwork_target(data_next_states).detach().max(1)[0].unsqueeze(1)
        data_Q_targets = data_rewards + (data_gamma * data_Q_targets_next * (1 - data_dones))
        data_Q_expected = self.data_qnetwork_local(data_states).gather(1, data_actions)
        data_loss = F.mse_loss(data_Q_expected, data_Q_targets)
        self.data_optimizer.zero_grad()
        data_loss.backward()
        self.data_optimizer.step()
        self.soft_update(self.data_qnetwork_local, self.data_qnetwork_target, TAU)                     

    def soft_update(self, data_local_model, data_target_model, data_tau):
        for i_target_param, i_local_param in zip(data_target_model.parameters(), data_local_model.parameters()):
            i_target_param.data.copy_(data_tau*i_local_param.data + (1.0-data_tau)*i_target_param.data)
    
    def step(self, data_state, data_action, data_reward, data_next_state, data_done):
        self.data_memory.add(data_state, data_action, data_reward, data_next_state, data_done)
        self.data_t_step = (self.data_t_step + 1) % UPDATE_EVERY
        if self.data_t_step == 0:
            if len(self.data_memory) > BATCH_SIZE:
                experiences = self.data_memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, data_state, data_eps=0.):
        data_state = torch.from_numpy(data_state).float().unsqueeze(0).to(device)
        self.data_qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.data_qnetwork_local(data_state)
        self.data_qnetwork_local.train()
        if random.random() > data_eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.data_action_size))


class ReplayBufferObject:
    def __init__(self, data_action_size, data_buffer_size, data_batch_size, data_seed):
        self.data_action_size = data_action_size
        self.data_memory = deque(maxlen=data_buffer_size)  
        self.data_batch_size = data_batch_size
        self.data_experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.data_seed = random.seed(data_seed)
    
    def sample(self):
        data_experiences = random.sample(self.data_memory, k=self.data_batch_size)

        data_states = torch.from_numpy(np.vstack([e.state for e in data_experiences if e is not None])).float().to(device)
        data_actions = torch.from_numpy(np.vstack([e.action for e in data_experiences if e is not None])).long().to(device)
        data_rewards = torch.from_numpy(np.vstack([e.reward for e in data_experiences if e is not None])).float().to(device)
        data_next_states = torch.from_numpy(np.vstack([e.next_state for e in data_experiences if e is not None])).float().to(device)
        data_dones = torch.from_numpy(np.vstack([e.done for e in data_experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (data_states, data_actions, data_rewards, data_next_states, data_dones)

    def __len__(self):
        return len(self.data_memory)
    
    def add(self, data_state, data_action, data_reward, data_next_state, data_done):
        data_e = self.data_experience(data_state, data_action, data_reward, data_next_state, data_done)
        self.data_memory.append(data_e)