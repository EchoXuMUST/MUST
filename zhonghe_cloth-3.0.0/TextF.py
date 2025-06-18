# -*- coding: utf-8 -*-
from paddleocr import PaddleOCR
import pytesseract
import logging

import cv2
import numpy as np
import os


def process_image(cv_image, rotate_flag=False):
    # 创建副本以避免修改原始图像
    img = cv_image.copy()
    original_region = []

    # 封装处理步骤为函数
    def process_and_find_boxes(image):
        # 灰度转换
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

        # Sobel边缘检测
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        # 二值化
        _, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        # 形态学操作
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 12))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 9))

        # 膨胀、腐蚀、再膨胀
        dilation = cv2.dilate(binary, element2, iterations=1)
        erosion = cv2.erode(dilation, element1, iterations=1)
        dilation2 = cv2.dilate(erosion, element2, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(dilation2, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_EXTERNAL cv2.RETR_TREE

        # 筛选区域
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 2000 < area < 2000000:  # 2048*2048/2 = 2,087,152
                # if area > 2000:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                regions.append(box)

        return regions

    # 处理原始图像
    original_region = process_and_find_boxes(img)
    rotated_region = []

    # 处理旋转后的图像（如果需要）
    if rotate_flag:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rotated_region = process_and_find_boxes(rotated_img)

        # 将旋转后的box转换回原始图像坐标
        h, w = img.shape[:2]
        # 创建逆向旋转矩阵
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1)

        transformed_boxes = []
        for box in rotated_region:
            # 将点转换为适当的格式
            points = box.reshape(4, 2).astype(np.float32)
            # 应用逆向旋转
            transformed_points = cv2.transform(points.reshape(1, -1, 2), M)[0]
            transformed_boxes.append(transformed_points.astype(int))

    else:
        transformed_boxes = []

    # 合并两个图像的区域
    region = original_region + transformed_boxes
    region = find_min_enclosing_boxes(region)

    # 合并相交的矩形
    region = merge_intersecting_boxes(region)

    return region


def find_min_enclosing_boxes(regions, distance_threshold=200):
    """
    找出距离相近的 box 并为每个分组计算最小外接矩形
    :param regions: 包含多个 box 的列表，每个 box 是一个 4x2 的 numpy 数组，表示矩形的四个顶点
    :param distance_threshold: 用于定义两个 box 是否相近的距离阈值
    :return: 一个列表，包含每个分组的最小外接矩形的四个顶点
    """
    # 初始化分组
    groups = []
    for box in regions:
        # 检查当前 box 是否可以加入已有的分组
        added = False
        for group in groups:
            # 计算当前 box 与分组中所有 box 的距离
            distances = np.linalg.norm(group - box, axis=2)
            if np.any(distances < distance_threshold):
                group.append(box)
                added = True
                break
        if not added:
            groups.append([box])

    # 为每个分组计算最小外接矩形
    min_enclosing_boxes = []
    for group in groups:
        # 将分组中的所有点合并
        group_points = np.vstack(group)
        # 计算最小外接矩形
        rect = cv2.minAreaRect(group_points)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        min_enclosing_boxes.append(box)

    return min_enclosing_boxes


def merge_intersecting_boxes(boxes):
    # 如果没有矩形，直接返回
    if not boxes:
        return []

    # 将box转换为边界框 [x, y, w, h]
    bounding_boxes = []
    for box in boxes:
        x_min = np.min(box[:, 0])
        y_min = np.min(box[:, 1])
        x_max = np.max(box[:, 0])
        y_max = np.max(box[:, 1])
        bounding_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    # 合并相交的矩形
    merged_boxes = [bounding_boxes[0]]

    for box in bounding_boxes[1:]:
        current_x, current_y, current_w, current_h = box
        merged = False

        for i in range(len(merged_boxes)):
            merged_x, merged_y, merged_w, merged_h = merged_boxes[i]

            # 计算交集区域
            intersection_x = max(0, min(current_x + current_w, merged_x + merged_w) - max(current_x, merged_x))
            intersection_y = max(0, min(current_y + current_h, merged_y + merged_h) - max(current_y, merged_y))
            intersection_area = intersection_x * intersection_y

            # 如果有交集，合并为一个矩形
            if intersection_area > 0:
                # 合并矩形
                new_x = min(current_x, merged_x)
                new_y = min(current_y, merged_y)
                new_w = max(current_x + current_w, merged_x + merged_w) - new_x
                new_h = max(current_y + current_h, merged_y + merged_h) - new_y
                merged_boxes[i] = (new_x, new_y, new_w, new_h)
                merged = True
                break

        # 如果没有与任何矩形相交，直接加入
        if not merged:
            merged_boxes.append(box)

    # 去掉在其他矩形内部的矩形
    final_boxes = []
    for i, box in enumerate(merged_boxes):
        is_contained = False
        for j, other_box in enumerate(merged_boxes):
            if i != j:
                # 检查是否在其他矩形内部
                if (box[0] >= other_box[0] and box[1] >= other_box[1] and
                        box[0] + box[2] <= other_box[0] + other_box[2] and
                        box[1] + box[3] <= other_box[1] + other_box[3]):
                    is_contained = True
                    break
        if not is_contained:
            final_boxes.append(box)

    # 将合并后的边界框转换回box格式
    merged_region = []
    for box in final_boxes:
        x, y, w, h = box
        merged_region.append(np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]))

    return merged_region


def detect_text_direction(np_img):
    """
    使用Canny边缘检测判断文本的主导方向，只计算前三条最长直线的角度
    """
    # 使用Canny边缘检测
    edges = cv2.Canny(np_img, 50, 150, apertureSize=3)

    # 定义结构元素
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 9))

    # 膨胀、腐蚀、再膨胀
    dilation = cv2.dilate(edges, element2, iterations=1)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(dilation, 1, np.pi / 180, threshold=30, minLineLength=60, maxLineGap=50)

    if lines is None:
        return 0

    # 计算每条直线的长度
    line_lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_lengths.append((length, line))

    # 按长度降序排序
    line_lengths.sort(reverse=True, key=lambda x: x[0])

    # 选择前三条最长的直线
    top_three_lines = line_lengths[:3]

    # 统计前三条最长直线的角度
    angles = []
    for _, line in top_three_lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    # 计算平均角度
    if angles:
        median_angle = np.median(angles)
    else:
        median_angle = 0

    return median_angle


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


def img2text_paddle(np_img):  # BGR_img,cv2
    # np_img = rotate_bound(np_img) # 图像旋转
    text = ""
    ocr = PaddleOCR(use_angle_cls=True,
                    use_gpu=True,
                    gpu_id=0,
                    show_log=False,
                    use_tensorrt=False,
                    lang="ch")
    result = ocr.ocr(np_img, cls=True)  # det=False
    if result is None:  # 检查 result 是否为 None
        print("No text detected.")
        return
    for idx in range(len(result)):
        res = result[idx]
        if res is None:  # 检查 res 是否为 None
            continue
        for line in res:
            if line is None:  # 检查 line 是否为 None
                continue
            if line[1][1] > 0:
                text += f"{line[1][0]},"
    print(text)
    return text


def img2text_tesseract(np_img):
    """
    对指定图像进行 OCR 识别，先识别原图，再旋转 180 度后识别一次。
    参数:
        np_img (numpy.ndarray): 图像的 NumPy 数组。
    返回:
        tuple: 包含两次识别结果的元组 (original_text, rotated_text)。
    """
    # 将图像从 BGR 转换为 RGB 格式（pytesseract 需要 RGB 格式）
    rgb_image = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    # 对原图进行 OCR 识别（指定中英文语言包）
    original_text = pytesseract.image_to_string(rgb_image, lang='chi_sim+eng')

    # 将图像旋转 180 度
    rotated_image = cv2.rotate(np_img, cv2.ROTATE_180)
    rgb_rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)

    # 对旋转后的图像进行 OCR 识别（指定中英文语言包）
    rotated_text = pytesseract.image_to_string(rgb_rotated_image, lang='chi_sim+eng')

    text = original_text + rotated_text
    print(text)
    return text


def rotate_bound(np_img):  # BGR_img,cv2
    img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    try:
        results = pytesseract.image_to_osd(img_rgb, config='--psm 0 -c min_characters_to_try=5',
                                           output_type=pytesseract.Output.DICT)
    except Exception as e:
        logging.warning("pytesseract.pytesseract.TesseractError：字符数过少，跳过旋转处理")
        return np_img

    angle = results["rotate"]
    if angle > 45:
        (h, w) = np_img.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(np_img, M, (nW, nH))
    else:
        return np_img