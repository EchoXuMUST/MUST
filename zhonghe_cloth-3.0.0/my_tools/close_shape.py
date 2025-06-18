import re
import os

from shapely import Polygon


class ShapeNode:
    def __init__(self, path, is_closed, min_x, max_x, min_y, max_y):
        self.path = path  # 图形路径
        self.is_closed = is_closed  # 是否封闭
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

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

    for path in paths_stack:
        if path:
            is_closed = is_closed_path(path)
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            shape = ShapeNode(path, is_closed, min_x, max_x, min_y, max_y)
            shapes.append(shape)

    return shapes

def find_closed_shapes(shapes):
    """找出所有的封闭图形"""
    return [shape for shape in shapes if shape.is_closed]

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
    closed_shapes = find_closed_shapes(shapes)
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
    print(len(root.children))
    return root

def generate_hpgl_instructions(shape, idx):
    """生成HPGL指令"""
    if not shape.path:
        return ""
    instructions = []
    instructions.append(f"; Shape {idx + 1}\n")  # 添加标识
    instructions.append("PU;")
    for i, point in enumerate(shape.path):
        x, y = point
        if i == 0:
            instructions.append(f"PU{int(x)},{int(y)};")
        else:
            instructions.append(f"PD{int(x)},{int(y)};")
    instructions.append("PU;")
    return "\n".join(instructions)

def save_to_plt_file(output_file, root):
    """将封闭图形的HPGL指令写入一个新的plt文件，仅输出子节点"""
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("IN;\n")
            file.write("SP;\n")  # 设置画笔为默认状态

            for idx, child in enumerate(root.children):
                file.write(generate_hpgl_instructions(child, idx))
                file.write("\n")

        print(f"成功将子节点图形写入文件 '{output_file}'。")
        return True
    except Exception as e:
        print(f"写入文件时发生错误：{e}")
        return False

def main():
    input_file = r"../example/CS24WB011M(84，104)大货[1]面_165-84+185-104 = 2套_1_144435.plt"  # 输入文件路径
    output_file = "output.plt"  # 输出文件路径

    content = read_plt_file(input_file)
    if content is None:
        return

    shapes = parse_hpgl_content(content)
    closed_shapes = [shape for shape in shapes if shape.is_closed]

    if not closed_shapes:
        print(f"文件 '{input_file}' 中没有找到封闭图形。")
        return

    root = build_tree(shapes)

    save_to_plt_file(output_file, root)

if __name__ == "__main__":
    main()