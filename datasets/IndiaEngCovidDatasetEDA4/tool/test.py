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


def calculate_balance(data):
    """计算每个类别的样本数量和需要增加的数据量"""
    # 统计每个类别的样本数量
    category_counts = {}
    for item in data:
        category = item["label"]
        category_counts[category] = category_counts.get(category, 0) + 1

    # 找到样本最多的类别数量
    max_count = max(category_counts.values()) if category_counts else 0

    # 计算每个类别需要增加的数量
    balance_info = {}
    for category, count in category_counts.items():
        balance_info[category] = {
            'current': count,
            'needed': max_count - count
        }

    return category_counts, balance_info

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

    category_counts, balance_info = calculate_balance(emotion_data)

    # 输出基本信息
    print(f"数据集中共有 {len(category_counts)} 个情感分类")
    print(f"样本总数: {len(emotion_data)}")

    # 输出每个类别的统计信息
    print("\n每个类别的样本数量和需要增加的数据量:")
    for i, (category, info) in enumerate(balance_info.items(), 1):
        print(f"{i}. {category}:")
        print(f"   - 当前样本数: {info['current']}")
        print(f"   - 需要增加: {info['needed']}")