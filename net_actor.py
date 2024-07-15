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
        self.fc2 = nn.Linear(256, 1)  # 输出，0 或 1
        self.fc3 = nn.Linear(256, 1)  # 输出，0 到 359
        self.fc4 = nn.Linear(256, 1)  # 输出，0 到 8
        self.fc5 = nn.Linear(256, 1)  # 输出0-10
        self.fc6 = nn.Linear(256, 1)  # 输出0-2
        self.fc7 = nn.Linear(256, 1)  # 输出0-356
        self.fc8 = nn.Linear(256, 1)  # 输出0-99
        self.fc9 = nn.Linear(256, 1)  # 输出0-4

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(self.fc1(x))
        move_action = torch.sigmoid(self.fc2(x))  # 使用 sigmoid 激活函数确保输出在 0 到 1 之间
        angle = torch.tanh(self.fc3(x)) * 179.5 + 179.5  # 使用 tanh 激活函数并缩放到 0 到 359
        info_action = torch.tanh(self.fc4(x)) * 4 + 4  # 使用 tanh 激活函数并缩放到 0 到 9

        attack_action = torch.tanh(self.fc5(x)) * 5 + 5  # 缩放到0-10
        action_type = torch.sigmoid(self.fc6(x)) * 2  # 缩放到0-2
        arg1 = torch.tanh(self.fc7(x)) * 232 + 232  # 缩放到0-464
        arg2 = torch.sigmoid(self.fc8(x)) * 99  # 缩放到0-99
        arg3 = torch.sigmoid(self.fc9(x)) * 4  # 缩放到0-4

        return move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3

