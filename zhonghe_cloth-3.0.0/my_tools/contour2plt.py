import json


def read_all_outer_contours_from_json(json_file_path, output_plt_file_path):
    """
    从JSON文件中读取所有Shape的Outer_contour，并将所有Shape的内容写入同一个.plt文件。

    :param json_file_path: JSON文件路径
    :param output_plt_file_path: 输出的.plt文件路径
    """
    try:
        # 从JSON文件中读取数据
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 打开输出的.plt文件
        with open(output_plt_file_path, 'w', encoding='utf-8') as plt_file:
            # 遍历所有Shape
            for shape_id, shape_data in data.items():
                # 获取Outer_contour
                outer_contour = shape_data.get("Outer_contour", "")

                # 写入Shape标识符
                plt_file.write(f"{shape_id}\n")

                # 写入Outer_contour内容
                plt_file.write(outer_contour + "\n")

        print(f"All outer contours have been saved to {output_plt_file_path}")

    except FileNotFoundError:
        print(f"Error: The file {json_file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file {json_file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")


# 示例用法

json_file_path = r'E:\code\project2025\cloth_dataset\data_process_v0513\cloth_json\CS24WB011M(84，104)大货[1]面_165-84+185-104 = 2套_1_144435.json'  # 替换为你的JSON文件路径
# json_file_path = "example.json"  # 替换为你的JSON文件路径
output_plt_file_path = "all_outer_contours.plt"  # 输出的.plt文件路径
read_all_outer_contours_from_json(json_file_path, output_plt_file_path)