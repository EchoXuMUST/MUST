import cv2
import numpy as np
import pytesseract
import logging
from pytesseract import Output


def rotate_bound(np_img):
    """
    检测图像中的文字方向，并根据需要对图像进行旋转纠偏

    参数:
        np_img: 输入的BGR格式的numpy图像数组

    返回:
        旋转后的图像
    """
    # 将BGR图像转换为RGB格式供pytesseract使用
    img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    try:
        # 使用pytesseract的OSD（Orientation and Script Detection）功能检测图像方向
        results = pytesseract.image_to_osd(img_rgb,
                                           config='--psm 0 -c min_characters_to_try=5',
                                           output_type=Output.DICT)
    except pytesseract.TesseractError as e:
        logging.warning("TesseractError: %s", str(e))
        logging.warning("字符数过少，无法准确检测文本方向，跳过旋转处理")
        return np_img

    # 获取检测到的旋转角度
    angle = results["rotate"]
    # confidence = results["confidence"]
    print(angle)

    # 根据检测到的角度和置信度判断是否需要旋转
    if angle > 45:  # 只有当置信度较高且角度较大时才进行旋转
        (h, w) = np_img.shape[:2]
        (cX, cY) = (w // 2, h // 2)  # 计算图像中心点

        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

        # 计算旋转后的新图像尺寸
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # 调整旋转矩阵以考虑图像尺寸变化
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # 对图像进行旋转
        rotated = cv2.warpAffine(np_img, M, (nW, nH))

        # 显示原始图像和旋转后的图像
        cv2.imshow('Original Image', np_img)
        # cv2.imshow('Rotated Image', rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return rotated
    else:
        # 显示原始图像
        cv2.imshow('Original Image', np_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # logging.info("无需旋转或旋转置信度不足。原始角度: %d, 置信度: %.2f", angle, confidence)
        return np_img


# 示例用法
if __name__ == "__main__":
    # 加载测试图像
    image = cv2.imread("./output2/text_1_1.png")

    # 进行文字方向检测和旋转
    rotated_image = rotate_bound(image)

    # 显示旋转后的图像
    # cv2.imshow('Final Rotated Image', rotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
