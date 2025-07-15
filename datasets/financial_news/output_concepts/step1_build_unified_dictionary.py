import os
import re
import json
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import hdbscan
import google.generativeai as genai

# --- 配置 ---
# 您提供的API密钥已直接填入
API_KEY = "AIzaSyDKWBfO2Eps4YQEhLJZaypxi4UJDD2s_7Y"

# 修改：提供所有包含原始概念的输入文件列表
# 您需要为您每个数据集（train, val, test）准备一个类似的文件
INPUT_FILES = ['concepts_summary_train.txt', 'concepts_summary_val.txt', 'concepts_summary_test.txt']

OUTPUT_CONCEPTS_DICT = 'standard_concepts.json'
OUTPUT_MAPPING = 'raw_to_standard_mapping.json'
ENCODER_MODEL = 'all-mpnet-base-v2'
MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = 1

# --- 初始化 Gemini API ---
if not API_KEY or API_KEY == "YOUR_API_KEY":
    print("错误：请提供一个有效的 Gemini API 密钥。")
    exit()
else:
    genai.configure(api_key=api_key)


def parse_raw_concepts_from_files(file_paths):
    """从多个文件中解析出所有独特的原始概念。"""
    print(f"正在从 {len(file_paths)} 个文件中解析原始概念...")
    raw_concepts = set()
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"警告: 输入文件 '{file_path}' 未找到，已跳过。")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 正则表达式匹配每个概念行
            concept_matches = re.findall(r'^\s*-\s*(.+)', content, re.MULTILINE)
            for concept in concept_matches:
                # 过滤掉无法生成的概念
                if "无法生成概念" not in concept:
                    raw_concepts.add(concept.strip())
        except Exception as e:
            print(f"读取或解析文件 '{file_path}' 时出错: {e}")

    print(f"成功从所有文件中提取到 {len(raw_concepts)} 个独特的原始概念。")
    return list(raw_concepts)


def get_standard_concept_name(model, cluster_concepts):
    """使用LLM为每个概念簇命名一个标准名称。"""
    concept_list_str = "\n".join(f"- {concept}" for concept in cluster_concepts)
    prompt = f"""
    你是一个金融术语专家。下面这个列表里的所有概念都指向同一个核心事件。请为这个事件提供一个最标准、最简洁、最中立的官方名称。

    要求:
    1. 名称应高度概括，例如将'layoffs', 'workforce reduction', 'job cuts'统一为'裁员与重组'。
    2. 只返回最终的标准名称，不要包含任何其他文字或解释。

    原始概念列表:
    {concept_list_str}

    标准概念名称:
    """
    try:
        time.sleep(2)  # 速率控制
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
        return response.text.strip()
    except Exception as e:
        print(f"  [!] LLM命名失败: {e}")
        return cluster_concepts[0]


def main():
    """主执行函数"""
    # 1. 解析所有文件中的原始概念
    raw_concepts = parse_raw_concepts_from_files(INPUT_FILES)
    if not raw_concepts:
        print("未找到任何概念，程序终止。")
        return

    # 2. 向量化所有概念
    print(f"正在加载句向量模型 '{ENCODER_MODEL}'...")
    encoder = SentenceTransformer(ENCODER_MODEL)
    print("开始向量化所有原始概念...")
    embeddings = encoder.encode(raw_concepts, show_progress_bar=True)

    # 3. 使用 HDBSCAN 进行聚类
    print("开始使用 HDBSCAN 进行概念聚类...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                                min_samples=MIN_SAMPLES,
                                metric='euclidean',
                                cluster_selection_method='eom')
    clusters = clusterer.fit_predict(embeddings)

    df = pd.DataFrame({'raw_concept': raw_concepts, 'cluster_id': clusters})
    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    noise_points = np.sum(clusters == -1)
    print(f"聚类完成。发现 {num_clusters} 个概念簇，以及 {noise_points} 个噪音点（未被聚类）。")

    # 4. LLM 对每个簇进行命名
    print("\n开始使用 Gemini API 为每个概念簇生成标准名称...")
    # 修正模型名称为有效的公开模型
    model = genai.GenerativeModel('gemini-1.5-flash')
    standard_concepts_map = {}

    for cluster_id in tqdm(range(num_clusters), desc="命名概念簇"):
        cluster_members = df[df['cluster_id'] == cluster_id]['raw_concept'].tolist()
        sample_members = cluster_members[:15]
        standard_name = get_standard_concept_name(model, sample_members)
        standard_concepts_map[cluster_id] = standard_name

    print("所有概念簇命名完成。")

    # 5. 生成最终产出文件
    raw_to_standard_mapping = {}
    for _, row in df.iterrows():
        cluster_id = row['cluster_id']
        if cluster_id == -1:
            standard_name = row['raw_concept']
        else:
            standard_name = standard_concepts_map[cluster_id]
        raw_to_standard_mapping[row['raw_concept']] = standard_name

    standard_concepts_dict = sorted(list(set(raw_to_standard_mapping.values())))

    # 6. 保存文件
    print(f"\n正在保存标准概念字典到 '{OUTPUT_CONCEPTS_DICT}'...")
    with open(OUTPUT_CONCEPTS_DICT, 'w', encoding='utf-8') as f:
        json.dump(standard_concepts_dict, f, ensure_ascii=False, indent=4)

    print(f"正在保存原始概念到标准概念的映射表到 '{OUTPUT_MAPPING}'...")
    with open(OUTPUT_MAPPING, 'w', encoding='utf-8') as f:
        json.dump(raw_to_standard_mapping, f, ensure_ascii=False, indent=4)

    print("\n第一步完成！")


if __name__ == '__main__':
    # 确保至少有一个输入文件存在
    if not any(os.path.exists(f) for f in INPUT_FILES):
        print("错误：在配置的 INPUT_FILES 中未找到任何输入文件。请检查文件路径。")
        # 为方便测试，创建一个示例文件
        print("正在创建一个示例 'concepts_summary_train.txt' 文件...")
        sample_content = "=== 类别: NEGATIVE (1个样本) ===\n--------------------------------------------------\n\n样本 1 (类别: negative):\n原始文本: In the Baltic countries , sales fell by 42.6 % .\n生成概念 (3个):\n  - sales decline\n  - economic downturn\n"
        with open('concepts_summary_train.txt', 'w', encoding='utf-8') as f:
            f.write(sample_content)
    main()
