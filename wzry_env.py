import itertools
import threading
import time

import torch

from globalInfo import GlobalInfo


class Environment():
    def __init__(self, android_controller, rewordUtil):
        self.android_controller = android_controller
        self.rewordUtil = rewordUtil
        self.globalInfo = GlobalInfo()
        self.lock = threading.Lock()


    def step(self, action):

        t_move_action, t_angle, t_info_action, t_attack_action, t_action_type, t_arg1, t_arg2, t_arg3 = action
        move_action = int(torch.round(t_move_action).item())
        angle = int(torch.round(t_angle).item())
        info_action = int(torch.round(t_info_action).item())
        attack_action = int(torch.round(t_attack_action).item())
        action_type = int(torch.round(t_action_type).item())
        arg1 = int(torch.round(t_arg1).item())
        arg2 = int(torch.round(t_arg2).item())
        arg3 = int(torch.round(t_arg3).item())

        self.android_controller.action_move({"action": move_action, "angle": angle})
        self.android_controller.action_info({"action": info_action})
        self.android_controller.action_attack(
            {"action": attack_action, "action_type": action_type, "arg1": arg1, "arg2": arg2, "arg3": arg3})


        next_state = self.android_controller.screenshot_window()
        while next_state is None or next_state.size == 0:
            time.sleep(0.01)
            next_state = self.android_controller.screenshot_window()
            continue

        reward, done, info = self.rewordUtil.get_reword(next_state, True, (move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3))

        return next_state, reward, done, info
