import datetime
import math
import random
import subprocess
import threading
import time
from queue import Queue, Empty

import cv2
import numpy as np

from argparses import move_actions_detail, info_actions_detail, attack_actions_detail, args


class AndroidTool:
    def __init__(self, scrcpy_dir="scrcpy-win64-v2.0"):
        self.scrcpy_dir = scrcpy_dir
        self.device_serial = args.device_id  # 修改为实际设备ID
        self.task_queue = Queue()
        self.threads = []
        self.stop_event = threading.Event()

        self.actual_height,self.actual_width = self.get_device_resolution()
        print(self.actual_width, self.actual_height)

        self._start_threads()

    def get_device_resolution(self):
        # 获取设备的实际分辨率
        output = subprocess.check_output(
            [f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell", "wm", "size"]
        ).decode('utf-8')
        resolution = output.split()[-1].split('x')
        return int(resolution[0]), int(resolution[1])


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
        # 移动逻辑
        action_index = task_params['action']
        if action_index == 1:
            actions_detail = move_actions_detail[action_index]
            start_x, start_y = self.calculate_startpoint(actions_detail['position'])

            end_x, end_y = self.calculate_endpoint((start_x, start_y),
                                        actions_detail['radius'],
                                        task_params['angle'])

            subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                            "input", "swipe", str(start_x), str(start_y), str(end_x), str(end_y), "500"])

    def execute_info(self, task_params):
        # 购买装备1， 购买装备2，发起进攻，开始撤退，请求集合，升级1技能，升级2技能，升级3技能，升级4技能
        action_index = task_params['action']
        if not action_index == 0:
            actions_detail = info_actions_detail[action_index]
            start_x, start_y = self.calculate_startpoint(actions_detail['position'])
            subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                            "input", "tap", str(start_x), str(start_y)])

    def execute_attack(self, task_params):
        # 点击操作逻辑
        action_index = task_params['action']
        action_type = task_params['action_type']
        # 角度
        arg1 = task_params['arg1']
        # 距离
        arg2 = task_params['arg2'] + 1
        # 长按时间
        arg3 = task_params['arg3'] + 1

        if action_index != 0:
            actions_detail = attack_actions_detail[action_index]
            start_x, start_y = self.calculate_startpoint(actions_detail['position'])
            if action_index < 7:
                subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                                "input", "tap", str(start_x), str(start_y)])
            else:
                if action_type == 0:
                    subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                                    "input", "tap", str(start_x), str(start_y)])
                elif action_type == 1:
                    end_x, end_y = self.calculate_endpoint((start_x, start_y),
                                                           arg2,
                                                           arg1)
                    subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                                    "input", "swipe", str(start_x), str(start_y), str(end_x), str(end_y), "500"])
                else:
                    subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                                    "input", "swipe", str(start_x), str(start_y), str(start_x), str(start_y), str(arg3 * 1000)])

    def calculate_startpoint(self, center):
        p_x, p_y = center
        start_x = int(self.actual_width * p_x)
        start_y = int(self.actual_height * p_y)
        return start_x, start_y

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

    def start_scrcpy(self):
        subprocess.Popen([f"{self.scrcpy_dir}/scrcpy.exe", "-s", self.device_serial, "-m", "1080", "--window-title", "wzry_ai"])

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
        self.task_queue.join()

    def take_screenshot(self):
        # Create a timestamp for the screenshot filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        screenshot_filename = f"screenshot_{timestamp}.png"

        try:
            # Take a screenshot using adb shell command
            result = subprocess.run([f'{self.scrcpy_dir}adb', 'exec-out', 'screencap', '-p'], capture_output=True, text=False)

            if result.returncode == 0:
                # Save the screenshot to a file
                with open(screenshot_filename, 'wb') as f:
                    f.write(result.stdout)
                print(f"Screenshot saved to {screenshot_filename}")
            else:
                print(f"Failed to take screenshot. Error: {result.stderr.decode('utf-8')}")
        except FileNotFoundError:
            print("adb is not installed or not found in your PATH.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def take_screenshot(self):
        try:
            # Take a screenshot using adb shell command
            result = subprocess.run([f'{self.scrcpy_dir}adb', 'exec-out', 'screencap', '-p'], capture_output=True, text=False)

            if result.returncode == 0:
                # Convert the screenshot to a numpy array
                screenshot_data = np.frombuffer(result.stdout, np.uint8)

                # Decode the image using OpenCV
                screenshot_image = cv2.imdecode(screenshot_data, cv2.IMREAD_COLOR)

                if screenshot_image is not None:
                    print("Screenshot captured successfully.")
                    return screenshot_image
                else:
                    print("Failed to decode the screenshot.")
            else:
                print(f"Failed to take screenshot. Error: {result.stderr.decode('utf-8')}")
        except FileNotFoundError:
            print("adb is not installed or not found in your PATH.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return None


def generate_random_number(n):
    return random.randint(0, n)


# 使用示例
if __name__ == "__main__":
    tool = AndroidTool()

    # 开始显示手机画面
    # tool.start_scrcpy()

    try:
        while True:
            # 随机调用方法示例
            tool.action_move({"action": 1, "angle": generate_random_number(359)})
            tool.action_info({"action": generate_random_number(8)})
            tool.action_attack({"action": generate_random_number(10),
                                "action_type": generate_random_number(2),
                                "arg1": generate_random_number(10),
                                "arg2": generate_random_number(99),
                                "arg3": generate_random_number(4)})

            time.sleep(0.01)  # 模拟随机间隔
    except KeyboardInterrupt:
        tool.stop()
