import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os


def split_dataset_to_json(file_path, val_size=0.15, test_size=0.15, random_state=42):
    """
    读取一个特定格式的CSV文件（第一列标签，第二列文本），
    将其划分为训练集、验证集和测试集，并分别保存为JSON Lines文件。
    输出的JSON对象将包含 'content' 和 'label' 键。
    """
    print(f"正在读取数据集: {file_path}")
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在。")
        return

    try:
        # --- 主要修改部分 (1/2) ---
        # 1. 读取CSV，并且直接指定列名
        # header=None -> 告诉pandas我们的CSV文件没有标题行
        # names=['label', 'content'] -> 指定第一列为'label'，第二列为'content'
        df = pd.read_csv(file_path, header=None, names=['label', 'content'], encoding='latin-1')
        print(f"数据集加载成功，总计 {len(df)} 条记录。")
        print("已将列重命名为: 'label' 和 'content'")

        # 用于分层抽样的列现在固定为 'label'
        stratify_col = 'label'
        stratify_data = df[stratify_col]
        print(f"将根据 '{stratify_col}' 列进行分层抽样。")
        # ------------------------

        # 第一次划分：分出训练集和临时集（验证集+测试集）
        remaining_size = val_size + test_size
        train_df, temp_df = train_test_split(
            df,
            test_size=remaining_size,
            random_state=random_state,
            stratify=stratify_data
        )

        # 为第二次划分准备分层数据
        stratify_temp = temp_df[stratify_col]

        # 第二次划分：从临时集中分出验证集和测试集
        relative_test_size = test_size / remaining_size
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=stratify_temp
        )

        print("\n划分完成:")
        print(f"  - 训练集大小: {len(train_df)} 条记录 ({len(train_df) / len(df) * 100:.0f}%)")
        print(f"  - 验证集大小: {len(val_df)} 条记录 ({len(val_df) / len(df) * 100:.0f}%)")
        print(f"  - 测试集大小: {len(test_df)} 条记录 ({len(test_df) / len(df) * 100:.0f}%)")

        # 构建输出文件名
        base_name = os.path.basename(file_path)
        name, _ = os.path.splitext(base_name)

        train_filename = f"{name}_train.jsonl"
        val_filename = f"{name}_validation.jsonl"
        test_filename = f"{name}_test.jsonl"

        # --- 主要修改部分 (2/2) ---
        # 重新排序列的顺序以满足 "content", "label" 的输出要求
        output_columns = ['content', 'label']
        train_df = train_df[output_columns]
        val_df = val_df[output_columns]
        test_df = test_df[output_columns]
        # ------------------------

        # 4. 将划分后的数据保存到新的JSON Lines文件中
        train_df.to_json(train_filename, orient='records', lines=True, force_ascii=False)
        print(f"\n训练集已保存至: {train_filename}")

        val_df.to_json(val_filename, orient='records', lines=True, force_ascii=False)
        print(f"验证集已保存至: {val_filename}")

        test_df.to_json(test_filename, orient='records', lines=True, force_ascii=False)
        print(f"测试集已保存至: {test_filename}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="将特定格式的CSV数据集划分为训练、验证和测试集，并输出为JSON Lines格式。")
    # parser.add_argument("file_path", type=str, default= "./all-data.csv",help="原始CSV文件的路径 (第一列标签，第二列文本)。")
    # parser.add_argument("--val_size", type=float, default=0.15, help="验证集所占的比例。")
    # parser.add_argument("--test_size", type=float, default=0.15, help="测试集所占的比例。")
    # parser.add_argument("--random_state", type=int, default=42, help="随机种子。")
    # args = parser.parse_args()

    split_dataset_to_json("./all-data.csv", 0.15, 0.15, 42)