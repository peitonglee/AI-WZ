import math
import random

import uiautomator2 as u2
import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty

from argparses import move_actions_detail, info_actions_detail, attack_actions_detail, args


class AndroidTool:
    def __init__(self, screenshot_size=1080):
        self.device = u2.connect(args.device_ip)
        self.screenshot_size = screenshot_size
        self.task_queue = Queue()
        self.threads = []
        self.stop_event = threading.Event()

        self._start_threads()

    def _worker(self, thread_id):
        while not self.stop_event.is_set():
            try:
                task_params = self.task_queue.get(timeout=1)
                if thread_id == 0 and task_params['action'] == 'move':
                    self.execute_move(task_params['params'])
                elif thread_id == 1 and task_params['action'] == 'attack':
                    self.execute_attack(task_params['params'])
                elif thread_id == 2 and task_params['action'] == 'info':
                    self.execute_info(task_params['params'])
                self.task_queue.task_done()
            except Empty:
                continue

    def _start_threads(self, num_threads=3):
        for i in range(num_threads):
            thread = threading.Thread(target=self._worker, args=(i,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def execute_move(self, task_params):
        # 无操作,移动逻辑
        action_index = task_params['action']
        if action_index == 1:
            actions_detail = move_actions_detail[action_index]
            start_x, start_y = actions_detail['position']

            end_x, end_y = self.calculate_endpoint(actions_detail['position'],
                                                   actions_detail['radius'],
                                                   task_params['angle'])
            self.device.swipe(start_x, start_y, end_x, end_y, duration=0.5)

    def execute_info(self, task_params):
        # 无操作, 购买装备1， 购买装备2，发起进攻，开始撤退，请求集合，升级1技能，升级2技能，升级3技能，升级4技能
        action_index = task_params['action']
        if not action_index == 0:
            actions_detail = info_actions_detail[action_index]
            start_x, start_y = actions_detail['position']

            self.device.click(start_x, start_y)

    def execute_attack(self, task_params):
        # 第三个线程的点击操作逻辑
        action_index = task_params['action']
        action_type = task_params['action_type']
        # 角度
        arg1 = task_params['arg1']
        # 距离
        arg2 = task_params['arg2'] + 1
        # 长按时间
        arg3 = task_params['arg3'] + 1

        if not action_index == 0:
            actions_detail = attack_actions_detail[action_index]
            start_x, start_y = actions_detail['position']
            if action_index < 7:
                self.device.click(start_x, start_y)
            else:
                # 点击，滑动，长按
                if action_type == 0:
                    self.device.click(start_x, start_y)
                elif action_type == 1:
                    end_x, end_y = self.calculate_endpoint(actions_detail['position'],
                                                           arg2,
                                                           arg1)
                    self.device.swipe(start_x, start_y, end_x, end_y, duration=0.5)
                else:
                    self.device.long_click(start_x, start_y, duration=arg3)

    def calculate_endpoint(self, center, radius, angle):
        """
        计算基于圆心、半径和角度的终点坐标。

        参数:
            center (tuple): 圆心坐标 (x, y)，通常是滑动开始的位置。
            radius (int): 从圆心到终点的距离。
            angle (int): 从x轴正方向顺时针旋转的角度，单位是度。

        返回:
            tuple: 终点坐标 (x, y)。

        坐标系说明:
            - 0度从x轴正方向开始（图形界面中，水平向右是x轴的正方向）。
            - 角度沿顺时针方向增加。
            - 90度位于y轴负方向（图形界面中，垂直向上是y轴的负方向）。
            - 180度位于x轴负方向（向左）。
            - 270度位于y轴正方向（图形界面中，垂直向下是y轴的正方向）。

        示例:
            为了计算从点 (100, 200) 开始，半径为 100，角度为 90度的终点位置：
            起始点为 x轴正方向，顺时针旋转 90度，将会指向屏幕的上方，
            结果终点坐标为 (100, 100)。
        """
        angle_rad = math.radians(angle)  # 将角度转换为弧度
        x = int(center[0] + radius * math.cos(angle_rad))
        y = int(center[1] + radius * math.sin(angle_rad))
        return (x, y)

    def action_move(self, params):
        self.task_queue.put({'action': 'move', 'params': params})

    def action_attack(self, params):
        self.task_queue.put({'action': 'attack', 'params': params})

    def action_info(self, params):
        self.task_queue.put({'action': 'info', 'params': params})

    def get_screenshot(self):
        screenshot = self.device.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        height, width = screenshot.shape[:2]
        if max(height, width) > self.screenshot_size:
            scale = self.screenshot_size / max(height, width)
            screenshot = cv2.resize(screenshot, (int(width * scale), int(height * scale)))
        return screenshot

    def start_screen_display(self):
        display_thread = threading.Thread(target=self._display_screen)
        display_thread.daemon = True
        display_thread.start()

    def _display_screen(self):
        while not self.stop_event.is_set():
            screenshot = self.get_screenshot()
            cv2.imshow('Android Screen', screenshot)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
        self.task_queue.join()


def generate_random_number(n):
    return random.randint(0, n)


# 使用示例
if __name__ == "__main__":
    tool = AndroidTool()

    # 开始显示手机画面
    tool.start_screen_display()

    try:
        while True:
            # 随机调用方法示例
            tool.action_move({"action": generate_random_number(1), "angle": generate_random_number(359)})
            tool.action_info({"action": generate_random_number(9)})

            tool.action_attack({"action": generate_random_number(10),
                                "action_type": generate_random_number(2),
                                "arg1": generate_random_number(10),
                                "arg2": generate_random_number(99),
                                "arg3": generate_random_number(4)})

            time.sleep(1)  # 模拟随机间隔
    except KeyboardInterrupt:
        tool.stop()
