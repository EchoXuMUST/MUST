import logging

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印

import warnings

warnings.filterwarnings("ignore")

import re
import os
import numpy as np
import cv2
import json
from tqdm import tqdm
from shapely.geometry import Polygon

import multiprocessing as mp

import TextF as TF
from my_tools.move_inter_lines import interfer_process
from my_tools.write_img import save_image_with_unique_name



class ShapeNode:
    def __init__(self, path, is_closed, min_x, max_x, min_y, max_y):
        self.path = path  # 图形路径
        self.is_closed = is_closed  # 是否封闭
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.children = []  # 子节点列表

def read_plt_file(file_path):
    """读取 .plt 文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if not content.strip():
            print(f"文件 '{file_path}' 存在，但内容为空。")
            return None
        print(f"成功读取文件 '{file_path}'。")
        return content
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在。")
        return None
    except UnicodeDecodeError:
        print(f"错误：文件 '{file_path}' 编码格式不是 UTF-8。")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None

def parse_hpgl_content(content):
    """解析 HPGL 内容，提取图形路径"""
    commands = re.findall(r'([A-Z]+)([^;]*);', content)
    shapes = []
    current_path = []
    paths_stack = []

    def is_closed_path(path):
        """判断路径是否封闭"""
        if not path:
            return False
        start_point = path[0]
        end_point = path[-1]
        return start_point == end_point

    for cmd, args in commands:
        if cmd == 'PU':
            points = re.findall(r'(\d+),(\d+)', args)
            points = [(int(x), int(y)) for x, y in points]
            if points:
                if current_path:
                    paths_stack.append(current_path.copy())
                current_path = points
        elif cmd == 'PD':
            points = re.findall(r'(\d+),(\d+)', args)
            points = [(int(x), int(y)) for x, y in points]
            if points:
                current_path.extend(points)
                if current_path:
                    is_closed = is_closed_path(current_path)
                    x_coords = [p[0] for p in current_path]
                    y_coords = [p[1] for p in current_path]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    shape = ShapeNode(current_path, is_closed, min_x, max_x, min_y, max_y)
                    shapes.append(shape)
                    current_path = []
    return shapes

def is_shape_inside_another(inner_shape, outer_shape, buffer=0):
    """判断一个图形是否在另一个封闭图形内部"""
    if not outer_shape.is_closed:  # 检查外形状是否为封闭路径
        return False
    if not inner_shape.path:  # 检查内形状是否有路径点
        return False

    inner_min_x, inner_max_x = inner_shape.min_x, inner_shape.max_x
    inner_min_y, inner_max_y = inner_shape.min_y, inner_shape.max_y

    outer_min_x, outer_max_x = outer_shape.min_x - buffer, outer_shape.max_x + buffer
    outer_min_y, outer_max_y = outer_shape.min_y - buffer, outer_shape.max_y + buffer

    if inner_min_x < outer_min_x or inner_max_x > outer_max_x or inner_min_y < outer_min_y or inner_max_y > outer_max_y:
        return False

    return True

def calculate_overlap_area(poly1, poly2):
    """计算两个多边形的重叠面积比例"""
    try:
        polygon1 = Polygon(poly1)
        polygon2 = Polygon(poly2)
        intersection_area = polygon1.intersection(polygon2).area
        union_area = polygon1.union(polygon2).area
        overlap_ratio = intersection_area / union_area if union_area != 0 else 0
        return overlap_ratio
    except Exception as e:
        print(f"Error in calculate_overlap_area: {e}")
        return 0

def build_tree(shapes):
    """构建树结构，确保有一个根节点包含所有封闭图形"""
    closed_shapes = [shape for shape in shapes if shape.is_closed]
    if not closed_shapes:
        return None

    root = max(closed_shapes, key=lambda s: (s.max_x - s.min_x) * (s.max_y - s.min_y))  # 找到面积最大的封闭图形作为根节点

    # 筛选子节点候选
    child_candidates = [shape for shape in closed_shapes if shape != root and is_shape_inside_another(shape, root)]

    # 去除重叠面积超过 90% 和包含在其他子节点内的子节点
    filtered_children = []
    for i, child in enumerate(child_candidates):
        # 判断是否重叠
        overlap = any(
            calculate_overlap_area(child.path, cj.path) > 0.9 for j, cj in enumerate(child_candidates) if j > i)
        if overlap:
            continue
        # 判断是否被其他子节点包含
        contained = any(is_shape_inside_another(child, other) for other in filtered_children)
        if not contained:
            filtered_children.append(child)

    root.children = filtered_children
    return root

def generate_hpgl_instructions(shape):
    """生成HPGL指令"""
    if not shape.path:
        return ""
    instructions = []
    instructions.append("PU;")
    for i, point in enumerate(shape.path):
        x, y = point
        if i == 0:
            instructions.append(f"PU{int(x)},{int(y)};")
        else:
            instructions.append(f"PD{int(x)},{int(y)};")
    instructions.append("PU;")
    return "\n".join(instructions)

def create_output_folder(folder_name):
    """创建输出文件夹"""
    try:
        os.makedirs(folder_name, exist_ok=True)
        print(f"创建文件夹 '{folder_name}' 成功。")
        return folder_name
    except Exception as e:
        print(f"创建文件夹时发生错误：{e}")
        return None


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


def filter_text(shape_commands, image, scales):

    def is_point_inside_polygon(point, polygon):
        """判断点是否在多边形内部（射线法）"""
        if not polygon:
            return False
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    ocr_text = ""

    old_scale, old_translatex, old_translatey, old_cs = scales
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cropped_regions = TF.process_image(gray_image, True)
    if not cropped_regions:
        print(f"没有识别到文字区域")
        return ocr_text

    # 遍历每个裁剪区域
    for region in cropped_regions:
        filtered_cmds = []
        # 过滤掉在裁剪区域外的指令
        for cmd in shape_commands.split('\n'):
            keep = True  # 假设指令保留
            if cmd.startswith('PU') or cmd.startswith('PD'):
                coords = list(map(int, re.findall(r'\d+', cmd)))
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * old_scale + old_translatex)
                    y = int(old_cs - (coords[i + 1] * old_scale + old_translatey))
                    # 如果有一个坐标不在区域内，就标记为不保留
                    if not is_point_inside_polygon([x, y], region.tolist()):
                        keep = False
                        break  # 一旦发现一个坐标不在区域内，就跳出循环
                if keep:
                    filtered_cmds.append(cmd)
                else:
                    filtered_cmds += f"PU;"

        final_filtered_cmds = "\n".join(filtered_cmds)
        text_image0, _ = normalize_and_draw_shape(f"{final_filtered_cmds}", canvas_size=512, thick=1)

        if text_image0 is not None:
            # 去除干扰线，这里去了两次，一次去不干净，感觉是细化线条和膨胀的问题
            text_image1 = interfer_process(text_image0)
            text_image2 = interfer_process(text_image1)

            # 图片文字转至水平
            angle = TF.detect_text_direction(text_image2)
            text_image = TF.deskew_image(text_image2, angle=angle)

            # 调试时使用，查看ocr的目标图片
            # save_image_with_unique_name("output3", text_image)

            # 利用飞浆识别文字
            ocr_text += TF.img2text_paddle(text_image)

    return ocr_text

def process_plt_files(ios):
    plt_file, output_folder, output_img_folder = ios
    new_file_name = os.path.basename(plt_file).split('.')[0]

    # folder_path = os.path.join(output_img_folder, new_file_name)
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # 读取和解析 plt 文件
    content = read_plt_file(plt_file)
    if content is None:
        return

    shapes = parse_hpgl_content(content)
    if not shapes:
        print(f"未识别到任何图形，跳过文件 {plt_file}。")
        return

    root = build_tree(shapes)
    if not root:
        print(f"未找到任何封闭图形，跳过文件 {plt_file}。")
        return

    # 提取所有非子节点图形
    non_child_shapes = []
    for shape in shapes:
        is_child = False
        for child in root.children:
            if (child.path == shape.path and child.is_closed == shape.is_closed and
                    child.min_x == shape.min_x and child.max_x == shape.max_x and
                    child.min_y == shape.min_y and child.max_y == shape.max_y):
                is_child = True
                break
        if not is_child:
            non_child_shapes.append(shape)

    all_shape = {}
    # 合并非子节点图形到对应的子节点，并生成HPGL指令和图片
    for i, child in enumerate(tqdm(root.children)):
        Outer_contour = generate_hpgl_instructions(child)
        Inner_lines = ""
        for shape in non_child_shapes:
            if is_shape_inside_another(shape, child, buffer=50):
                Inner_lines += "\n" + generate_hpgl_instructions(shape)

        # entire_shape = Outer_contour + Inner_lines
        image, scales = normalize_and_draw_shape(f"{Inner_lines}", canvas_size=2048, thick=1,
                                                 outline_commends=Outer_contour, text_only=True)

        # 部件图片保存
        # image2, _ = normalize_and_draw_shape(f"{Inner_lines}", canvas_size=1024, thick=1,
        #                                      outline_commends=Outer_contour)
        # output_path = os.path.join(str(folder_path), f'Shape{i}.jpg')
        # print(output_path)
        # cv2.imencode('.jpg', image2)[1].tofile(output_path)  # 保存循环图片text_image

        if image is not None:
            ocr_text = filter_text(f"{Inner_lines}", image, scales)
        else:
            ocr_text = ""

        all_shape[f"Shape{i}"] = dict(Outer_contour=Outer_contour, Inner_lines=Inner_lines, ocr_text=ocr_text)

    # 将结果保存为 JSON 文件
    file_path = os.path.join(output_folder, new_file_name + ".json")
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(all_shape, json_file, ensure_ascii=False, indent=4)

    print(f"字典已成功写入到: {file_path}")



if __name__ == "__main__":
    # input_folder = r"E:\code\project2025\cloth_dataset\24年plt打印文件"
    input_folder = "example"
    output_folder = './output2'
    output_img_folder = './output3'

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        output_folder = create_output_folder(output_folder)

    plt_files = []

    # 获取output_folder中所有文件的文件名（不包含后缀）
    output_files = set(os.path.splitext(filename)[0] for filename in os.listdir(output_folder))

    # 遍历输入文件夹中的所有 .plt 文件
    for root, dirs, files in os.walk(input_folder):
        for plt_file in files:
            if plt_file.endswith(".plt"):
                file_name_without_ext = plt_file[:-4]  # 获取名文件（不包含后缀）
                if file_name_without_ext not in output_files:
                    plt_files.append(os.path.join(root, plt_file))
    print(f"总共待转换plt文件数：{len(plt_files)}")

    for plt_file in tqdm(plt_files):
        if not plt_file.endswith('.plt'):
            continue

        process_plt_files([plt_file, output_folder, output_img_folder])

    # 并行化处理
    # num_processes = 8  # mp.cpu_count()
    # with mp.Pool(processes=num_processes) as pool:
    #     args = [(plt_file, output_folder, output_img_folder) for plt_file in plt_files]
    #     list(tqdm(pool.imap(process_plt_files, args), total=len(plt_files)))