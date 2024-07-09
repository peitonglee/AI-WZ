import torch
import torch.nn as nn
import torch.nn.functional as F

from net_feature import NetFeature


# Actor 网络
class NetActor(nn.Module):
    def __init__(self, action_dim):
        super(NetActor, self).__init__()
        self.feature_extractor = NetFeature()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(self.fc1(x))
        action = torch.tanh(self.fc2(x))
        return action
