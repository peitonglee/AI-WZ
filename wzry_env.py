import itertools
import threading
import time

import torch

from globalInfo import GlobalInfo
from methodutil import conver_model_result_to_action


class Environment():
    def __init__(self, android_controller, rewordUtil):
        self.android_controller = android_controller
        self.rewordUtil = rewordUtil
        self.globalInfo = GlobalInfo()
        self.lock = threading.Lock()


    def step(self, action, is_ppo):

        if is_ppo:
            t_move_action, t_angle, t_info_action = action
            move_action = int(torch.round(t_move_action).item())
            angle = int(torch.round(t_angle).item())
            info_action = int(torch.round(t_info_action).item())
            print(move_action, angle, info_action)

            self.android_controller.action_move({"action": move_action, "angle": angle})
            self.android_controller.action_info({"action": info_action})


        else:
            t_attack_action, t_action_type, t_arg1, t_arg2, t_arg3 = action
            attack_action = int(torch.round(t_attack_action).item())
            action_type = int(torch.round(t_action_type).item())
            arg1 = int(torch.round(t_arg1).item())
            arg2 = int(torch.round(t_arg2).item())
            arg3 = int(torch.round(t_arg3).item())
            self.android_controller.action_attack({"action": attack_action,
                                "action_type": action_type,
                                "arg1": arg1,
                                "arg2": arg2,
                                "arg3": arg3})


        # 执行动作
        start_time = time.time()
        next_state = self.android_controller.take_screenshot()
        while next_state is None:
            time.sleep(0.01)
            next_state = self.android_controller.take_screenshot()
            continue

        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        reward, done, info = self.rewordUtil.get_reword(next_state, True)

        return next_state, reward, done, info
