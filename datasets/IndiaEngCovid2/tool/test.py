import json


# 1. 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# 2. 统计分类数量
def count_categories(data):
    # 使用集合(Set)自动去重
    unique_categories = set()

    # 遍历每个条目
    for item in data:
        category = item["label"]
        unique_categories.add(category)

    return unique_categories


# 3. 主程序
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    json_file_path = "../train.json"

    # 读取数据
    emotion_data = read_json_file(json_file_path)

    # 统计分类
    categories = count_categories(emotion_data)

    # 输出结果
    print(f"数据集中共有 {len(categories)} 个情感分类")
    print("具体分类如下:")
    for i, category in enumerate(sorted(categories), 1):
        print(f"{i}. {category}")