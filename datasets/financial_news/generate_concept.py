import google.generativeai as genai
import os
import json
import time
from collections import defaultdict

# --- 配置您的 API 密钥 ---
# 您提供的API密钥已直接填入
# 为了安全，建议您未来使用环境变量或密钥管理服务来存储密钥
api_key = "AIzaSyDKWBfO2Eps4YQEhLJZaypxi4UJDD2s_7Y"

if not api_key or api_key == "YOUR_API_KEY":
    print("错误：请提供一个有效的 Gemini API 密钥。")
else:
    genai.configure(api_key=api_key)


def parse_samples_by_label(file_path):
    """
    从 JSON 文件中解析样本，并按 'label' 分类。
    现在使用 'content' 作为文本的键。
    """
    samples_by_label = defaultdict(list)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if 'content' in item and 'label' in item:
                samples_by_label[item['label']].append(item['content'])
        return samples_by_label
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是一个有效的JSON文件。")
        return None
    except Exception as e:
        print(f"读取或解析JSON文件时出错: {e}")
        return None


def generate_concepts_for_sample(model, sample_text):
    """
    为单个样本生成5个核心概念。
    """
    prompt = f"""
    你是一名专业的金融分析师。请阅读以下这条财经新闻，用5个简短、精确、中立的短语总结其核心的商业或金融'概念'。

    要求：
    1. 生成5个不同的概念。
    2. 每个概念都应该高度概括，抓住事件的本质。
    3. 每个概念都应该是一个常见的商业或金融术语/事件。
    4. 每个概念占一行，前面不要加任何数字或符号。

    新闻文本：
    "{sample_text}"

    5个核心概念：
    """
    try:
        # 使用较低的 temperature 来获取更稳定和一致的输出
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.3))
        concepts = [concept.strip().lstrip('- ') for concept in response.text.strip().split('\n') if concept.strip()]
        return concepts
    except Exception as e:
        # 捕获并打印API错误信息，特别是速率限制错误
        print(f"为样本 '{sample_text[:40]}...' 生成概念时出错: {e}")
        return []


def main():
    """
    主执行函数
    """
    input_file = './train.json'
    output_file = 'concept_analysis_report.txt'

    print(f"开始从 '{input_file}' 中按类别解析样本...")
    samples_by_label = parse_samples_by_label(input_file)
    if not samples_by_label:
        print("未能解析到任何样本，程序终止。")
        return

    for label, sentences in samples_by_label.items():
        print(f"找到类别 '{label}' 的样本 {len(sentences)} 条。")
    print("-" * 40)

    # 注意：根据您的代码，您似乎想使用 gemini-2.0-flash，但目前这不是一个有效的公开模型名称。
    # 我将使用 'gemini-1.5-flash'，这是一个高效且强大的模型。如果您有特定模型的访问权限，请修改此处。
    model = genai.GenerativeModel('gemini-2.0-flash')

    all_concepts_by_label = defaultdict(list)

    print("开始为每个样本生成5个核心概念...")

    # 计算总样本数，用于进度显示
    total_samples = sum(len(s) for s in samples_by_label.values())
    processed_samples = 0
    # 按标签排序以保证处理和输出顺序
    for label, sentences in sorted(samples_by_label.items()):
        print(f"\n--- 正在处理类别: {label.upper()} ---")
        sentences = sentences[:3]
        for i, sentence in enumerate(sentences):
            processed_samples += 1
            print(f"  处理样本 {processed_samples}/{total_samples} (类别: {label})...")

            concepts = generate_concepts_for_sample(model, sentence)

            if concepts:
                all_concepts_by_label[label].extend(concepts)

            # 速率控制：每分钟20条，即每条请求后等待 60/20 = 3 秒
            time.sleep(3)

    print("\n所有样本的概念生成完毕。")
    print("-" * 40)

    # --- 直接输出，无需精简 ---
    final_report_content = []
    report_title = "按情感类别划分的概念清单"
    final_report_content.append(report_title)
    final_report_content.append("=" * len(report_title))

    for label, concepts in sorted(all_concepts_by_label.items()):
        final_report_content.append(f"\n\n### {label.capitalize()} 概念 (Concepts for '{label}')\n")
        # 为每个概念添加项目符号
        for concept in concepts:
            final_report_content.append(f"- {concept}")

    # 将最终报告写入 TXT 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(final_report_content))
        print(f"\n报告已成功写入到文件: '{output_file}'")
    except Exception as e:
        print(f"\n写入文件时出错: {e}")


if __name__ == '__main__':
    # 确保您的 train.json 文件存在于脚本同目录下
    # 如果没有，创建一个示例文件，并使用 'content'作为键
    if not os.path.exists('./train.json'):
        print("未找到 'train.json' 文件，正在创建一个示例文件。")
        sample_data = [
            {"content": "The contract covers the supply of temporary heating equipment for LKAB 's new pellet plant.",
             "label": "neutral"},
            {"content": "Operating profit surged to EUR21m from EUR106,000.", "label": "positive"},
            {"content": "The company's net sales fell by 5% from the previous accounting period.", "label": "negative"}
        ]
        with open('./train.json', 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)

    main()