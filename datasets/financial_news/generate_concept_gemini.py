import google.generativeai as genai
import os
import json
import time
from tqdm import tqdm

# --- 安全配置 Gemini API ---
# 警告：请不要将 API 密钥直接写入代码中。
# 建议使用环境变量，这样更安全。
# 从环境变量中读取 API 密钥
api_key = os.getenv("GEMINI_API_KEY")
print(api_key)
if not api_key:
    # 如果您不想设置环境变量，可以临时在此处填入密钥，但请注意安全风险：
    # api_key = "YOUR_API_KEY_HERE"
    raise ValueError("请设置 GEMINI_API_KEY 环境变量。出于安全考虑，不建议将密钥硬编码在脚本中。")

genai.configure(api_key=api_key)

# --- 定义模型和提示 ---

# 用于概念提取的提示模板
PROMPT_TEMPLATE = """
You are a professional financial analyst, please read this text and summarize the connotation of this sample with up to one to six short, precise and neutral financial concept phrases, which can be used as the basis for classifying this sample.

Requirement: 
1. Generate up to 1 to 6 different concepts. 
2. Each concept should be highly summarized to grasp the essence of the event. 
3. Each concept should be a common business or financial term/event. 
4. Each concept should occupy one line and no numbers or symbols should be added before it. 
5. The output concept should be associated with the category of the sample. 
6. The meanings of the output concepts should not be repetitive. 
7. If the sample text is too short or incomplete, please output "Concept cannot be generated".

The output must be a valid JSON list of strings. For example: ["Concept A", "Concept B", "Concept C"].
If no relevant concepts are found, return an empty list [].

Financial News Text:
"{text}"
Label：
"{label}"

Standardized Concepts (JSON list):
"""


def get_concepts_from_gemini(text_content: str,text_label: str, max_retries: int = 3) -> list:
    """
    使用 Gemini API 从文本中提取概念。

    Args:
        text_content: 输入的金融新闻文本。
        max_retries: 最大重试次数。

    Returns:
        提取出的概念列表，如果失败则返回空列表。
    """
    # 注意：'gemini-2.0-flash' 不是一个有效的模型名称。
    # 这里我们使用 'gemini-1.5-flash'，它是一个速度快且能力强的模型，适合此类任务。
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = PROMPT_TEMPLATE.format(text=text_content,label=text_label)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)

            # 清理并解析模型的输出
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()

            concepts = json.loads(cleaned_response)

            if isinstance(concepts, list):
                return sorted(list(set(concepts)))
            else:
                print(f"警告: API返回的不是一个列表: {concepts}")
                return []

        except json.JSONDecodeError:
            print(f"\n错误: 无法解析API的JSON输出。重试次数 {attempt + 1}/{max_retries}")
            print(f"原始输出: {response.text}")
            time.sleep(2 ** attempt)
        except Exception as e:
            # 捕获并打印更详细的API错误信息
            print(f"\n调用API时发生错误: {e}。重试次数 {attempt + 1}/{max_retries}")
            time.sleep(2 ** attempt)

    print(f"\n错误: 达到最大重试次数后仍然失败。文本: '{text_content[:80]}...'")
    return []


def process_dataset_realtime(input_path: str, output_path: str):
    """
    处理整个数据集文件，并实时将结果输出到JSON文件和控制台。
    """
    print(f"正在读取文件: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {input_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件格式不正确，无法解析JSON: {input_path}")
        return

    # 根据您的脚本，文本键是 'content'
    text_key = 'content'
    label_key = 'label'
    print(f"处理开始，结果将实时写入: {output_path}")

    # 以写入模式打开文件，准备实时追加
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write('[\n')  # 写入JSON数组的起始括号

        total_samples = len(dataset)
        # 使用 enumerate 来追踪样本索引，以便正确处理逗号
        for i, sample in enumerate(tqdm(dataset, desc=f"正在处理 {os.path.basename(input_path)}")):
            if text_key not in sample:
                print(f"\n警告: 样本中未找到键 '{text_key}'，已跳过。样本: {sample}")
                continue

            text = sample.get(text_key, "")
            label = sample.get(label_key, "")
            # 调用API获取概念
            concepts = get_concepts_from_gemini(text,label)

            # 创建新的样本，包含原始数据和概念
            new_sample = sample.copy()
            new_sample['concepts'] = concepts

            # --- 实时输出 ---
            # 1. 实时打印到控制台
            # 使用 print 代替 tqdm.write 可以在新行中清晰地打印多行JSON
            console_output = json.dumps(new_sample, indent=2, ensure_ascii=False)
            print(
                f"\n--- 样本 {i + 1}/{total_samples} 处理完成 ---\n{console_output}\n---------------------------------")

            # 2. 实时写入JSON文件
            # 将字典转为格式化的JSON字符串
            json_string = json.dumps(new_sample, indent=4, ensure_ascii=False)

            # 如果不是第一个元素，先写入一个逗号和换行符
            if i > 0:
                f_out.write(',\n')

            f_out.write(json_string)
            f_out.flush()  # 确保内容立即写入磁盘

        f_out.write('\n]\n')  # 写入JSON数组的结束括号

    print(f"全部处理完成！最终结果已完整保存在 {output_path}")


if __name__ == "__main__":
    # 定义输入和输出文件路径
    train_input_file = os.path.join("./", "train.json")
    train_output_file = "concept_train.json"

    val_input_file = os.path.join("./", "val.json")
    val_output_file = "concept_val.json"

    # 使用新的实时处理函数处理训练集
    process_dataset_realtime(train_input_file, train_output_file)

    # 处理验证集
    process_dataset_realtime(val_input_file, val_output_file)