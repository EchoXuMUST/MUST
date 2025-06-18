import json
import re
import cv2
import numpy as np



def normalize_and_draw_shape(commends, canvas_size=2048, thick=1, outline_commends="", text_only=False):
    """标准化并绘制图形"""
    points = []
    if text_only:
        scarl_commands = outline_commends
        shape_commands = commends
    else:
        scarl_commands = outline_commends + commends
        shape_commands = outline_commends + commends
    for cmd in scarl_commands.split('\n'):
        if cmd.startswith('PU') or cmd.startswith('PD'):
            coords = list(map(int, re.findall(r'\d+', cmd)))
            points.extend([(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)])

    if not points:
        print("not points")
        return None, None

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        print("width <= 0 or height <= 0")
        return None, None

    scale = min(canvas_size / width, canvas_size / height) * 0.95
    translate_x = (canvas_size - (max_x - min_x) * scale) / 2 - min_x * scale
    translate_y = (canvas_size - (max_y - min_y) * scale) / 2 - min_y * scale

    # 创建白色背景的图像
    image = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    current_point = None
    for cmd in shape_commands.split('\n'):
        if cmd.startswith('PU'):
            coords = list(map(int, re.findall(r'\d+', cmd)))
            if coords:
                # 调整坐标
                x = int(coords[0] * scale + translate_x)
                y = int(canvas_size - (coords[1] * scale + translate_y))
                current_point = (x, y)
            else:
                current_point = None
        elif cmd.startswith('PD'):
            coords = list(map(int, re.findall(r'\d+', cmd)))
            translated_points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * scale + translate_x)
                y = int(canvas_size - (coords[i + 1] * scale + translate_y))
                translated_points.append((x, y))
            if current_point:
                # 绘制线条
                for i in range(len(translated_points)):
                    cv2.line(image, current_point, translated_points[i], (0, 0, 0), thick)
                    current_point = translated_points[i]
            else:
                if len(translated_points) >= 2:
                    # 绘制多段线
                    for i in range(len(translated_points) - 1):
                        cv2.line(image, translated_points[i], translated_points[i + 1], (0, 0, 0), thick)
                    current_point = translated_points[-1]

    scales = [scale, translate_x, translate_y, canvas_size]

    return image, scales



def read_plt_file(file_path):
    """
    从PLT文件中读取绘图指令（PU和PD命令）
    :param file_path: PLT文件路径
    :return: 包含指令的字符串，每条指令用换行符分隔
    """
    commands = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 清理行内容并保留有效指令
                cleaned_line = line.strip().upper()
                if cleaned_line.startswith(('PU', 'PD')):
                    commands.append(cleaned_line)
        return '\n'.join(commands)
    except Exception as e:
        print(f"读取PLT文件出错: {e}")
        return ''


# 示例用法
if __name__ == "__main__":
    # 从文件中读取主要绘图指令和轮廓指令
    # main_commands = read_plt_file(r"E:\code\project2025\cloth_dataset\11.plt")

    file_path = r'E:\code\project2025\cloth_dataset\data_process_v0513\cloth_json\CS24WB011M(84，104)大货[1]面_165-84+185-104 = 2套_1_144435.json'  # 替换为你的JSON文件路径
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 将JSON文件内容加载为Python字典

    shape_number = 33
    # 提取Shape4相关的数据
    shape_data = data.get(f"Shape{shape_number}")
    if shape_data is None:
        print("错误：JSON中未找到 'Shape4' 键。")
    else:
        outer_contour = shape_data.get("Outer_contour")
        inner_lines = shape_data.get("Inner_lines")
        ocr_text = shape_data.get("ocr_text")


    # 生成图像
    image, scales = normalize_and_draw_shape(
        commends=inner_lines,
        canvas_size=2048,
        thick=1,
        text_only=False
    )

    # 显示结果
    if image is not None:
        cv2.imwrite("output_image.png", image)
        # cv2.imshow("Drawn Shape", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()