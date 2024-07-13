import itertools
import threading
import time

from globalInfo import GlobalInfo
from methodutil import conver_model_result_to_action


class Environment():
    def __init__(self, android_controller, rewordUtil):
        self.android_controller = android_controller
        self.rewordUtil = rewordUtil
        self.globalInfo = GlobalInfo()
        self.lock = threading.Lock()

        # Action1: 361种可能 (0: 无操作, 1-360: 滑动角度)
        action1_space = list(range(361))

        # Action2: 15种可能 (0: 无操作, 1-14: 其他点击事件)

        action2_space = list(range(15))

        # Action3: 包含 无操作 召唤师技能，1技能，2技能，3技能
        # 召唤师技能，1技能，2技能，3技能有: 点击，滑动，长按
        # 滑动有滑动角度360度，和滑动距离 距离最大100
        # 长按有长按时间 最大5s

        # 无操作, 购买装备1， 购买装备2，发起进攻，开始撤退，请求集合，升级1技能，升级2技能，升级3技能，升级4技能

        # 无操作, 攻击，攻击小兵，攻击塔，回城，恢复，装备技能, 1技能，2技能，3技能,
        # 点击，滑动，长按
        # 360度 滑动距离 长按时间
        action3_space = []

        # 无操作
        action3_space.append(0)

        # 召唤师技能，1技能，2技能，3技能
        action3_skill = [0, 1, 2, 3]
        # 点击，滑动，长按
        action3_event = [0, 1, 2]

        for skill in action3_skill:
            for event in action3_event:
                if event == 0:  # 点击
                    action3_space.append((skill, event))
                elif event == 1:  # 滑动
                    for angle in range(361):
                        for distance in range(101):
                            action3_space.append((skill, event, angle, distance))
                elif event == 2:  # 长按
                    for duration in range(501):  # 长按时间从0到500（表示0到5秒）
                        action3_space.append((skill, event, duration / 100))

        print(action3_space)

        # 组合动作空间[(action1, action2，action3), .... ,(action1, action2，action3)]
        self.action_space = list(itertools.product(action1_space, action2_space, action3_space))

        self.action_space_n = len(self.action_space)

        print(self.action_space_n)


def step(self, action):
    real_action = conver_model_result_to_action(action)
    self.android_controller.execute_actions(real_action)

    next_state = self.globalInfo.get_global_frame()
    while next_state is None:
        time.sleep(0.01)
        next_state = self.globalInfo.get_global_frame()
        continue

    reward, done, info = self.rewordUtil.get_reword(next_state, True)

    return next_state, reward, done, info
