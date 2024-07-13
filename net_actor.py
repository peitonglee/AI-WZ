import torch
import torch.nn as nn
import torch.nn.functional as F

from net_feature import NetFeature


# Actor 网络
class NetActor(nn.Module):
    def __init__(self):
        super(NetActor, self).__init__()
        self.feature_extractor = NetFeature()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)  # 分类输出，0 或 1
        self.fc3 = nn.Linear(256, 1)  # 回归输出，0 到 359
        self.fc4 = nn.Linear(256, 1)  # 回归输出，0 到 9

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(self.fc1(x))
        move_action = torch.sigmoid(self.fc2(x))  # 使用 sigmoid 激活函数确保输出在 0 到 1 之间
        angle = torch.tanh(self.fc3(x)) * 179.5 + 179.5  # 使用 tanh 激活函数并缩放到 0 到 359
        info_action = torch.tanh(self.fc4(x)) * 4.5 + 4.5  # 使用 tanh 激活函数并缩放到 0 到 9
        return move_action, angle, info_action
