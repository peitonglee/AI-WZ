import torch
import torch.nn as nn
import torch.nn.functional as F


# Actor 网络
class NetActor(nn.Module):
    def __init__(self):
        super(NetActor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)

        conv_output_size = self._get_conv_output_size(640, 640)
        self.fc = nn.Linear(conv_output_size, 256)

        self.fc1 = nn.Linear(256, 256)
        self.fc_move = nn.Linear(256, 2)  # move_action_list
        self.fc_angle = nn.Linear(256, 360)  # angle_list
        self.fc_info = nn.Linear(256, 9)  # info_action_list
        self.fc_attack = nn.Linear(256, 11)  # attack_action_list
        self.fc_action_type = nn.Linear(256, 3)  # action_type_list
        self.fc_arg1 = nn.Linear(256, 360)  # arg1_list
        self.fc_arg2 = nn.Linear(256, 100)  # arg2_list
        self.fc_arg3 = nn.Linear(256, 5)  # arg3_list
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

        move_action = F.softmax(self.fc_move(x), dim=-1)
        angle = F.softmax(self.fc_angle(x), dim=-1)
        info_action = F.softmax(self.fc_info(x), dim=-1)
        attack_action = F.softmax(self.fc_attack(x), dim=-1)
        action_type = F.softmax(self.fc_action_type(x), dim=-1)
        arg1 = F.softmax(self.fc_arg1(x), dim=-1)
        arg2 = F.softmax(self.fc_arg2(x), dim=-1)
        arg3 = F.softmax(self.fc_arg3(x), dim=-1)

        return move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3
