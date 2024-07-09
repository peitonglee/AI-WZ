import torch
import torch.nn as nn
import torch.nn.functional as F

from net_feature import NetFeature


# Critic 网络
class NetCritic(nn.Module):
    def __init__(self, action_dim):
        super(NetCritic, self).__init__()
        self.feature_extractor = NetFeature()
        self.fc1 = nn.Linear(256 + action_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state, action):
        state = self.feature_extractor(state)
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value
