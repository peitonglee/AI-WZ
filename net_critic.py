import torch
import torch.nn as nn
import torch.nn.functional as F


# Critic 网络
class NetCritic(nn.Module):
    def __init__(self):
        super(NetCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)

        conv_output_size = self._get_conv_output_size(640, 640)
        self.fc = nn.Linear(conv_output_size, 256)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)

        self._initialize_weights()

    def _get_conv_output_size(self, height, width):
        dummy_input = torch.zeros(1, 3, height, width)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
        return x.view(x.size(0), -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value
