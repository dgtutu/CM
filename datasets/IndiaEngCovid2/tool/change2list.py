import re


def extract_from_txt(file_path):
    """
    从txt文件中提取<example>标签内容并返回列表

    参数:
        file_path (str): txt文件路径

    返回:
        list: 包含所有标签内容的Python列表
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 提取所有标签内容
        pattern = r'<example>(.*?)</example>'
        items = re.findall(pattern, content)

        return items

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到")
        return []
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return []


# 使用示例
if __name__ == "__main__":
    # 替换为你的txt文件路径
    txt_file = "../train.txt"

    # 执行提取
    result_list = extract_from_txt(txt_file)

    # 打印结果
    print("提取的列表内容:")
    for i, item in enumerate(result_list, 1):
        print(f"{i}. {item}")

    print("\n完整列表:")
    print(result_list)