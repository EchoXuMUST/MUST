import os
import cv2

# for box in region:
#     cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

def save_image_with_unique_name(output_dir, img):
    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'processed_image.png')
    # 检查文件是否存在，如果存在则添加唯一标识符
    if os.path.exists(output_path):
        output_path = os.path.join(output_dir, f'processed_image_{str(0)}.png')
        counter = 0
        while os.path.exists(output_path):
            counter += 1
            output_path = os.path.join(output_dir, f'processed_image_{str(counter)}.png')
    cv2.imwrite(output_path, img)   # 保存图像
    print(f"Image saved to: {output_path}")
    return output_path  # 可以选择返回保存的路径