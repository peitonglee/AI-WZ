import math

import cv2
import numpy as np

# 读取本地图片文件
image_path = "screenshot_2024-07-15_05-07-34.png"
image = cv2.imread(image_path)


def calculate_endpoint(center, radius, angle):
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

# 检查图片是否成功加载
if image is None:
    print("Error: Could not open or find the image.")
else:
    # 获取图片的宽度和高度
    height, width, _ = image.shape

    c_x = int(width * 0.844)
    c_y = int(height * 0.58)

    # print(c_y)
    #
    # x, y = calculate_endpoint((c_x, c_y), 200, 0)


    # 定义圆点的中心坐标和半径
    center_coordinates = (c_x, c_y)
    radius = 20
    color = (0, 255, 0)  # 绿色
    thickness = -1  # 填充整个圆

    # 在图片上画圆点
    cv2.circle(image, center_coordinates, radius, color, thickness)

    # 显示图片
    cv2.imshow("Image with Circle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()