import threading
import time
import numpy as np
import scrcpy

import argparses
from androidController import AndroidController
from android_tool import AndroidTool
from getReword import GetRewordUtil
from globalInfo import GlobalInfo
from ppo_agent import PPO_Agent

from wzry_env import Environment
from onnxRunner import OnnxRunner

# 全局状态
globalInfo = GlobalInfo()

class_names = ['started']
start_check = OnnxRunner('models/start.onnx', classes=class_names)

ppo_agent = PPO_Agent()


rewordUtil = GetRewordUtil()
tool = AndroidTool()
env = Environment(tool, rewordUtil)


def main():
    return_list = []
    epoch = 0
    state = None
    next_state = None

    while True:
        # 获取当前的图像
        state = tool.take_screenshot()
        # 保证图像能正常获取
        if state is None:
            time.sleep(0.01)
            continue
        # cv2.imwrite(f"tmp/img_0.jpg", state)
        # 初始化对局状态 对局未开始
        globalInfo.set_game_end()
        # 判断对局是否开始
        checkGameStart = start_check.get_max_label(state)

        if checkGameStart == 'started':
            print("-------------------------------对局开始-----------------------------------")
            globalInfo.set_game_start()

            # 这一局的总回报
            epoch_return_total = 0
            epsilon = 0
            # 对局开始了，进行训练
            while globalInfo.is_start_game():
                # 获取预测动作
                action, move_action, angle, info_action = ppo_agent.select_action(state)
                globalInfo.set_value("action", action)

                next_state, reward, done, info = env.step((move_action, angle, info_action), True)
                print(info)


                # 对局结束
                if done == 1:
                    epsilon = epsilon + 1
                    print("-------------------------------对局结束-----------------------------------")
                    globalInfo.set_game_end()
                    print(f"Episode: {epoch}, Reward total: {epoch_return_total},  Time: {time}, Epsilon: {epsilon}")
                    break

                # 追加经验
                globalInfo.store_transition_ppo(ppo_agent.preprocess_image(state), action, reward, ppo_agent.preprocess_image(next_state), done)

                state = next_state

                epoch_return_total += reward

            # 保存每一局结束的reword
            return_list.append(epoch_return_total)
            # 计算前n个元素的平均值
            average = np.mean(return_list[:epoch])
            print("average reword", average)
            epoch = epoch + 1

        else:
            print("对局未开始")
            time.sleep(0.1)



if __name__ == '__main__':
    ppo_agent.start_train()
    training_thread = threading.Thread(target=main)
    training_thread.start()
    training_thread.join()