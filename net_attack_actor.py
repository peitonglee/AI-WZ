import torch
import torch.nn as nn
import torch.nn.functional as F

from net_feature import NetFeature


# Actor 网络
class NetAttackActor(nn.Module):
    def __init__(self):
        super(NetAttackActor, self).__init__()
        self.feature_extractor = NetFeature()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)  # 输出0-10
        self.fc3 = nn.Linear(256, 1)  # 输出0-2
        self.fc4 = nn.Linear(256, 1)  # 输出0-356
        self.fc5 = nn.Linear(256, 1)  # 输出0-99
        self.fc6 = nn.Linear(256, 1)  # 输出0-4

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(self.fc1(x))
        attack_action = torch.tanh(self.fc2(x)) * 5 + 5  # 缩放到0-10
        action_type = torch.sigmoid(self.fc3(x)) * 2  # 缩放到0-2
        arg1 = torch.tanh(self.fc4(x)) * 232 + 232  # Scale to 0-464
        arg2 = torch.sigmoid(self.fc5(x)) * 99  # Scale to 0-99
        arg3 = torch.sigmoid(self.fc6(x)) * 4  # Scale to 0-4
        return attack_action, action_type, arg1, arg2, arg3
