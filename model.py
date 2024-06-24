import torch
import torch.nn as nn
import torch.nn.functional as F

class ModuleQNetwork(nn.Module):
    def __init__(self, data_state_size, data_action_size, data_seed, data_fc1_units=128, data_fc2_units=128):
        super(ModuleQNetwork, self).__init__()
        self.data_seed = torch.manual_seed(data_seed)
        self.data_fc1 = nn.Linear(data_state_size, data_fc1_units)
        self.data_fc2 = nn.Linear(data_fc1_units, data_fc2_units)
        self.data_fc3 = nn.Linear(data_fc2_units, data_action_size)

    def forward(self, data_state):
        data_x = F.relu(self.data_fc1(data_state))
        data_x = F.relu(self.data_fc2(data_x))
        return self.data_fc3(data_x)