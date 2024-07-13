import cv2
import numpy as np


def cal(img):
    # 获取图像的尺寸
    image_height, image_width = img.shape[:2]

    # 截取矩形的固定宽度和高度
    width = image_width * 0.116
    height = image_height * 0.024

    total_area = int(width * height)

    # 计算中心顶部矩形的起始点
    # left = int(image_width * 0.568)
    left = int(image_width * 0.57)
    top = int(image_height * 0.019)  # 从顶部开始
    right = int(left + width)
    bottom = int(top + height)

    # 根据计算出的坐标裁剪图像
    cropped_img = img[top:bottom, left:right]

    # 将图片从BGR转换到HSV色彩空间
    hsv_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # 定义BGR颜色 #AF363E
    bgr_color = np.uint8([[[62, 54, 175]]])  # 注意这里是BGR格式
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
    hue = hsv_color[0][0][0]

    # 设置颜色范围的容错率
    tolerance = 30  # 容差值可以根据需要调整

    # 定义HSV中想要提取的颜色范围
    lower_bound = np.array([hue - tolerance, 50, 50])
    upper_bound = np.array([hue + tolerance, 255, 255])

    # 使用cv2.inRange()函数找到图像中颜色在指定范围内的区域
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # 将掩码应用于原图像，只保留指定颜色的区域
    color_segment = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

    # 转成灰度图
    gray = cv2.cvtColor(color_segment, cv2.COLOR_BGR2GRAY)

    # 找到指定颜色的最右边的位置
    rightmost_position = 0
    for col in range(gray.shape[1]):
        if np.any(gray[:, col] != 0):
            rightmost_position = col

    # 计算指定颜色的面积
    area = (rightmost_position + 1) * height

    print(int((area * 100) / total_area))  # 百分比计算，去掉小数部分

    cv2.imshow('Image', img)
    # 使用OpenCV显示裁剪后的图像
    cv2.imshow('Cropped Image', gray)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口


# 使用OpenCV读取图像
img = cv2.imread("H:\\video\\monster\\train2\\attackMonster_chicken\\video4_video4_frame_0004741.jpg")
cal(img)
