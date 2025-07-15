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
INPUT_FILE = 'concepts_summary.txt'
OUTPUT_CONCEPTS_DICT = 'standard_concepts.json'
OUTPUT_MAPPING = 'raw_to_standard_mapping.json'
# Sentence Transformer 模型，用于将概念文本转换为向量
ENCODER_MODEL = 'all-mpnet-base-v2'
# HDBSCAN 聚类参数
MIN_CLUSTER_SIZE = 3  # 一个簇中最少的概念数量
MIN_SAMPLES = 1  # 决定一个点成为核心点的邻域数量

# --- 初始化 Gemini API ---
if not API_KEY or API_KEY == "YOUR_API_KEY":
    print("错误：请提供一个有效的 Gemini API 密钥。")
    exit()
else:
    genai.configure(api_key=API_KEY)


def parse_raw_concepts(file_path):
    """从 concepts_summary.txt 中解析出所有独特的原始概念。"""
    print(f"正在从 {file_path} 中解析原始概念...")
    raw_concepts = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 正则表达式匹配每个概念行
        concept_matches = re.findall(r'^\s*-\s*(.+)', content, re.MULTILINE)
        for concept in concept_matches:
            # 过滤掉无法生成的概念
            if "无法生成概念" not in concept:
                raw_concepts.add(concept.strip())
        print(f"成功提取到 {len(raw_concepts)} 个独特的原始概念。")
        return list(raw_concepts)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{file_path}' 未找到。")
        return []


def get_standard_concept_name(model, cluster_concepts):
    """使用LLM为每个概念簇命名一个标准名称。"""
    # 将簇内概念格式化为字符串列表
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
        # 添加速率控制
        time.sleep(2)  # 每次调用前等待2秒
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
        return response.text.strip()
    except Exception as e:
        print(f"  [!] LLM命名失败: {e}")
        # 如果API失败，返回第一个概念作为临时名称
        return cluster_concepts[0]


def main():
    """主执行函数"""
    # 1. 解析原始概念
    raw_concepts = parse_raw_concepts(INPUT_FILE)
    print(raw_concepts)

if __name__ == '__main__':
    main()
