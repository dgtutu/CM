import os
import re
import json
from tqdm import tqdm

# --- 配置 ---
# 将输入文件和输出文件配对
DATA_SPLITS = {
    'train': {'input': 'concepts_summary_train.txt', 'output': 'cbm_training_data_train.jsonl'},
    'val': {'input': 'concepts_summary_val.txt', 'output': 'cbm_training_data_val.jsonl'},
    'test': {'input': 'concepts_summary_test.txt', 'output': 'cbm_training_data_test.jsonl'}
}
INPUT_MAPPING_FILE = 'raw_to_standard_mapping.json'


def parse_samples_with_raw_concepts(file_path):
    """从 concepts_summary.txt 解析出样本、标签和原始概念列表。"""
    samples_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 正则表达式匹配每个完整的样本块
        sample_blocks = re.findall(r'样本 \d+ \(类别: (.*?)\):\n原始文本: (.*?)\n生成概念.*?:\n((?:\s*-\s*.*\n?)*)',
                                   content)

        for block in sample_blocks:
            label = block[0].strip().lower()
            text = block[1].strip()
            raw_concepts_str = block[2]
            raw_concepts = [c.strip() for c in raw_concepts_str.strip().split('- ') if c.strip()]
            raw_concepts = [c for c in raw_concepts if "无法生成概念" not in c]

            if text and label and raw_concepts:
                samples_data.append({
                    "text": text,
                    "label": label,
                    "raw_concepts": raw_concepts
                })
        return samples_data
    except FileNotFoundError:
        print(f"错误: 输入文件 '{file_path}' 未找到。")
        return []


def main():
    """主执行函数"""
    # 1. 加载统一的映射文件
    print(f"正在加载映射文件 '{INPUT_MAPPING_FILE}'...")
    try:
        with open(INPUT_MAPPING_FILE, 'r', encoding='utf-8') as f:
            raw_to_standard_mapping = json.load(f)
    except FileNotFoundError:
        print(f"错误: 映射文件 '{INPUT_MAPPING_FILE}' 未找到。请先运行 step1_build_unified_dictionary.py。")
        return

    # 2. 遍历每个数据切片 (train, val, test)
    for split_name, paths in DATA_SPLITS.items():
        input_path = paths['input']
        output_path = paths['output']

        print(f"\n--- 正在处理数据切片: {split_name} ---")

        if not os.path.exists(input_path):
            print(f"警告: 输入文件 '{input_path}' 未找到，跳过此切片。")
            continue

        # 3. 解析包含原始概念的样本数据
        samples_data = parse_samples_with_raw_concepts(input_path)
        if not samples_data:
            continue

        # 4. 生成该切片的训练数据文件
        print(f"正在为 '{split_name}' 生成训练数据文件...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in tqdm(samples_data, desc=f"处理 {split_name} 样本"):
                # 使用映射表将原始概念转换为标准概念
                standard_concepts = sorted(list(set(
                    raw_to_standard_mapping.get(raw_concept, raw_concept) for raw_concept in item['raw_concepts']
                )))

                training_item = {
                    "text": item['text'],
                    "label": item['label'],
                    "standard_concepts": standard_concepts
                }
                f.write(json.dumps(training_item, ensure_ascii=False) + '\n')

        print(f"数据切片 '{split_name}' 的训练数据已保存到 '{output_path}'。")

    print("\n第二步完成！")


if __name__ == '__main__':
    main()
