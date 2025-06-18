import cv2
import numpy as np
import sys

# 增加最大递归深度
sys.setrecursionlimit(10000)

# 1. 图像预处理：二值化和细化
def preprocess_image(gray_image):
    # 二值化
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 细化
    # 定义结构元素
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # 膨胀、腐蚀、再膨胀
    dilation = cv2.dilate(binary_image, element2, iterations=1)
    thinned_image = cv2.ximgproc.thinning(dilation)
    return thinned_image

# 2. 构建基本图形
class BasicNode:
    def __init__(self, position):
        self.position = position
        self.type = None
        self.connections = []

def build_basic_graph(thinned_image):
    height, width = thinned_image.shape
    graph = np.zeros((height, width), dtype=object)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(height):
        for x in range(width):
            if thinned_image[y, x] == 255:
                node = BasicNode((x, y))
                node.type = 0  # 初始类型为孤立节点
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and thinned_image[ny, nx] == 255:
                        node.type += 1
                        node.connections.append((nx, ny))
                graph[y, x] = node
    return graph

# 3. 检测主曲线
def detect_main_curves(graph, angle_threshold=1.0, length_threshold=100):
    height, width = graph.shape
    visited = np.zeros((height, width), dtype=bool)
    main_curves = []

    def dfs(node, path, prev_angle=None):
        if visited[node.position[1], node.position[0]]:
            return
        visited[node.position[1], node.position[0]] = True
        path.append(node.position)
        is_path_ended = True  # 标记路径是否结束
        for nx, ny in node.connections:
            next_node = graph[ny, nx]
            if next_node and not visited[ny, nx]:
                if len(path) > 1:
                    # 计算当前节点与前一个节点的方向
                    x1, y1 = path[-2]
                    x2, y2 = path[-1]
                    current_angle = np.arctan2(y2 - y1, x2 - x1)
                    if prev_angle is not None and np.abs(current_angle - prev_angle) > angle_threshold:
                        # 方向变化超过阈值，检查路径长度
                        if len(path) >= length_threshold:
                            main_curves.append(path.copy())
                        # 重置路径并继续搜索剩余路径
                        path = [node.position]  # 从当前节点重新开始
                        prev_angle = None  # 重置角度
                    else:
                        prev_angle = current_angle
                dfs(next_node, path, prev_angle)
                is_path_ended = False
        if is_path_ended and len(path) >= length_threshold:
            # 如果路径搜索到头且长度满足阈值，加入 main_curves
            main_curves.append(path.copy())

    for y in range(height):
        for x in range(width):
            node = graph[y, x]
            if node and not visited[y, x] and node.type == 1:  # 从端点开始
                path = []
                dfs(node, path)
    return main_curves

# 4. 显示图像
def show_image(image, window_name="image", width=2048, height=2048):
    """
    显示图像，并调整窗口大小。
    :param image: 要显示的图像
    :param window_name: 窗口名称
    :param width: 窗口宽度
    :param height: 窗口高度
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 5. 去除干扰线
def interfer_process(image):
    # 读取图像,并转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    # 图像预处理
    thinned_image = preprocess_image(gray)
    # 构建基本图形
    basic_graph = build_basic_graph(thinned_image)
    # 检测主曲线
    main_curves = detect_main_curves(basic_graph)

    # 可视化结果
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for curve in main_curves:
        for x, y in curve:
            cv2.circle(result_image, (x, y), 1, (255, 255, 255), -1)  # 用红色标记检测到的干扰线

    # show_image(result_image, "Result")

    return result_image


if __name__ == "__main__":
    image_path = r"E:\code\project2025\cloth_data_process\output5\processed_image_57.png"
    # image_path = "output_image.png"
    image = cv2.imread(image_path)
    interfer_process(image)