import os
import json
import time
from collections import defaultdict
from openai import OpenAI

# --- 配置您的 DeepSeek API 密钥 ---
api_key = "sk-"
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

MAX_PROCESS_NUM = -1
def parse_samples_by_label(file_path):
    """
    从 JSON 文件中解析样本，并按 'label' 分类。
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


def generate_concepts_for_sample(sample_text,label):
    """
    为单个样本生成5个核心概念。
    """
    prompt_sys = """
    你是一名专业的金融分析师。请阅读该财经新闻，用最多6个简短、精确、中立的英文概念短语总结该样本的内涵，用于作为该样本的分类的依据。

    要求：
    1. 生成最多1-6个不同的概念。
    2. 每个概念都应该高度概括，抓住事件的本质。
    3. 每个概念都应该是一个常见的商业或金融术语/事件。
    4. 每个概念占一行，前面不要加任何数字或符号。
    5. 输出的概念应与样本的类别相关联。
    6. 输出的概念间的含义不要重复。
    7. 如果样本文本过短或不完整，请输出"无法生成概念"。
    """

    prompt_user = f"""
    新闻文本：
    "{sample_text}"
    该样本的类别是：
    "{label}"
    请输出1-6个核心概念：
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.3,
            max_tokens=500
        )

        # 解析响应
        content = response.choices[0].message.content
        concepts = [concept.strip() for concept in content.split('\n') if concept.strip()]
        return concepts[:5]  # 返回最多5个概念
    except Exception as e:
        print(f"API请求出错: {e}")
        return []


def save_concepts_for_label(label, concept, output_dir="output"):
    """
    为特定标签保存概念列表到文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"concepts_{label}.txt")

    # 写入文件
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\n".join(concept))
            f.write("\n")  # 添加换行符以分隔不同样本的概念
        print(f"已生成类别 '{label}' 的概念文件: {output_file}")
    except Exception as e:
        print(f"写入类别 '{label}' 文件时出错: {e}")


def main():
    """
    主执行函数
    """
    start_time = time.time()
    input_file = './train.json'
    output_dir = 'output_concepts'
    all_results = defaultdict(list)
    print(f"开始从 '{input_file}' 中按类别解析样本...")
    samples_by_label = parse_samples_by_label(input_file)
    if not samples_by_label:
        print("未能解析到任何样本，程序终止。")
        return

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: '{output_dir}'")


    processed_samples = 0

    print("开始为每个样本生成5个核心概念...")

    # 按标签处理，每个类别最多处理3个样本
    for label, sentences in sorted(samples_by_label.items()):
        print(f"\n--- 正在处理类别: {label.upper()} ---")
        # label_concepts = []  # 存储当前类别的所有概念
        # 计算总样本数（每个类别最多处理3个样本）
        total_samples = len(sentences)
        # 处理当前类别的样本（最多3个）
        for sentence in sentences[:MAX_PROCESS_NUM]:
            processed_samples += 1
            print(f"  处理样本 {processed_samples}/{total_samples} (类别: {label})...")

            concepts = generate_concepts_for_sample(sentence,label)

            if concepts:
                print(f"    生成概念: {', '.join(concepts)}")
                # label_concepts.extend(concepts)
                # 保存当前类别的结果
                # if label_concepts:
                save_concepts_for_label(label, concepts, output_dir)
                # 将样本和概念存储起来用于生成汇总文件
                all_results[label].append({  # 添加到结果字典
                    "text": sentence,  # 原始文本
                    "concepts": concepts  # 生成的概念列表
                })
            # time.sleep(1)  # 稍微等待一下，避免频繁请求

        processed_samples = 0
        print(f"类别 '{label.upper()}' 处理完成")
        print("-" * 40)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(time_elapsed)
    print("\n所有类别的概念生成完毕。")

    # 生成详细汇总文件（包含每个样本的文本和概念）
    try:
        summary_path = os.path.join(output_dir, "concepts_summary.txt")  # 汇总文件路径
        with open(summary_path, 'w', encoding='utf-8') as f:  # 打开汇总文件
            f.write("各情感类别概念清单详细汇总\n")  # 写入标题
            f.write("=" * 40 + "\n\n")  # 标题下划线

            total_samples = 0  # 重置样本计数器

            for label, samples in all_results.items():  # 遍历每个标签的结果
                if samples:  # 检查是否有样本
                    f.write(f"=== 类别: {label.upper()} ({len(samples)}个样本) ===\n")  # 写入类别标题
                    f.write("-" * 50 + "\n")  # 类别分隔线

                    for i, sample in enumerate(samples, 1):  # 遍历类别中的每个样本
                        total_samples += 1  # 增加总样本计数
                        f.write(f"\n样本 {i} (类别: {label}):\n")  # 样本编号
                        f.write(f"原始文本: {sample['text']}\n")  # 原始文本
                        f.write(f"生成概念 ({len(sample['concepts'])}个):\n")  # 概念数量

                        for concept in sample['concepts']:  # 遍历每个概念
                            f.write(f"  - {concept}\n")  # 以项目符号形式写入概念

                    f.write("\n\n")  # 样本间空行

            f.write("\n" + "=" * 50 + "\n")  # 结束分隔线
            f.write(f"汇总完成! 总共处理了 {total_samples} 个样本\n")  # 完成统计

        print(f"已生成详细汇总文件: {summary_path}")  # 成功通知
    except Exception as e:  # 处理汇总文件错误
        print(f"生成汇总文件时出错: {e}")


if __name__ == '__main__':
    # 确保train.json文件存在

    if not os.path.exists('./train.json'):
        print("未找到 'train.json' 文件，正在创建示例文件。")
        sample_data = [
            {"content": "公司公布年度财报显示净利润增长20%", "label": "positive"},
            {"content": "新产品发布因安全问题被监管部门叫停", "label": "negative"},
            {"content": "公司任命新首席执行官", "label": "neutral"},
            {"content": "公司宣布新产品上市，预计将增加30%市场份额", "label": "positive"},
            {"content": "公司因环保问题被罚款200万美元", "label": "negative"},
            {"content": "公司总部迁至新的办公大楼", "label": "neutral"},
            {"content": "与主要竞争对手达成战略合作协议", "label": "positive"},
            {"content": "主要供应商宣布停止供货", "label": "negative"},
            {"content": "公司发布季度财务报告", "label": "neutral"}
        ]
        with open('./train.json', 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)

    main()