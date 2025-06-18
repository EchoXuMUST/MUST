
# image_path = "../output3/processed_image_0.png"  # 替换为你的图像路径

import cv2
import numpy as np


def detect_text_direction(image):
    """
    使用Canny边缘检测判断文本的主导方向
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        print("未检测到直线，无法判断文本方向")
        return 0, image

    # 统计所有直线的角度
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    # 计算平均角度
    avg_angle = np.mean(angles)

    # 打印检测到的平均角度
    print(f"检测到的文本平均角度: {avg_angle:.2f}度")

    return avg_angle, edges


def deskew_image(image, angle):
    """
    根据检测到的角度对图像进行倾斜校正
    """
    # 获取图像的中心点
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 旋转图像
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


# 读取图像
image_path = "../output3/processed_image_0.png"  # 替换为你的图像路径
image = cv2.imread(image_path)

if image is None:
    print("无法加载图像，请检查路径")
else:
    # 检测文本方向
    angle, edges = detect_text_direction(image)

    # 如果检测到的角度不为零，则进行倾斜校正
    if angle != 0:
        rotated_image = deskew_image(image, 45)
        print(f"校正后的文本角度: {angle:.2f}度")

        # 显示校正后的图像
        cv2.imshow("Deskewed Image", rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("图像已水平，无需校正")