import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
import os

# 导入您自定义的模型
from modules import RobertaCBL

# --- 配置 ---
# 数据文件路径
TRAIN_FILE = 'cbm_training_data_train.jsonl'
VAL_FILE = 'cbm_training_data_val.jsonl'
TEST_FILE = 'cbm_training_data_test.jsonl'
CONCEPTS_DICT_FILE = 'standard_concepts.json'

# 模型保存路径
MODEL_G_PATH = 'concept_encoder_distilled.pth'
MODEL_F_PATH = 'label_predictor.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'

# 报告输出路径
GLOBAL_REPORT_PATH = 'cbm_global_analysis_report.txt'
TEST_PREDICTIONS_PATH = 'test_set_predictions.txt'

# 模型与训练参数
# 学生模型 (我们要训练的模型)
STUDENT_MODEL_NAME = 'roberta-base'
# 老师模型 (用于生成“真值”激活度)
TEACHER_MODEL_NAME = 'all-mpnet-base-v2'

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5  # 知识蒸馏可能需要更多轮次
LEARNING_RATE = 2e-5
DROPOUT = 0.1
TOP_K_CONCEPTS = 5


# --- 1. 数据加载与准备 ---

def load_data(filepath):
    """加载JSONL格式的数据"""
    if not os.path.exists(filepath):
        print(f"警告: 文件 '{filepath}' 未找到。将返回空列表。")
        return []
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


class DistillationDataset(Dataset):
    """为知识蒸馏创建PyTorch数据集"""

    def __init__(self, data, tokenizer, teacher_model, teacher_concept_vectors):
        self.data = data
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model
        self.teacher_concept_vectors = teacher_concept_vectors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 为学生模型(RobertaCBL)准备输入
        student_encoding = self.tokenizer.encode_plus(
            item['text'],
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # 使用老师模型(SentenceTransformer)为当前样本生成“真值”激活度
        # 这是一个向量，每一维代表样本与对应概念的相似度
        with torch.no_grad():
            sample_embedding_teacher = self.teacher_model.encode(item['text'], convert_to_tensor=True)
            ground_truth_activations = util.cos_sim(sample_embedding_teacher, self.teacher_concept_vectors).squeeze()

        return {
            'input_ids': student_encoding['input_ids'].flatten(),
            'attention_mask': student_encoding['attention_mask'].flatten(),
            'ground_truth_activations': ground_truth_activations,
            'label': item['label']
        }


# --- 2. 训练与评估流程 ---

def train_cbm(train_data, val_data, concept_dict):
    """训练CBM模型 g (RobertaCBL)，使用知识蒸馏"""
    print("\n--- 开始训练概念编码器 (g)，采用知识蒸馏 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化学生模型 (RobertaCBL)
    student_tokenizer = RobertaTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    student_model = RobertaCBL(concept_dim=len(concept_dict), dropout=DROPOUT).to(device)

    # 初始化老师模型 (SentenceTransformer)
    print(f"正在加载老师模型: {TEACHER_MODEL_NAME}")
    teacher_model = SentenceTransformer(TEACHER_MODEL_NAME, device=device)

    # 使用老师模型预计算概念向量
    print("使用老师模型预计算'真值'概念向量...")
    with torch.no_grad():
        teacher_concept_vectors = teacher_model.encode(concept_dict, convert_to_tensor=True)

    # 创建数据集
    train_dataset = DistillationDataset(train_data, student_tokenizer, teacher_model, teacher_concept_vectors)
    val_dataloader = None
    if val_data:
        val_dataset = DistillationDataset(val_data, student_tokenizer, teacher_model, teacher_concept_vectors)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 使用余弦相似度损失，让学生模型的输出向量与老师模型的激活度向量尽可能相似
    loss_fn = torch.nn.CosineEmbeddingLoss()

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        student_model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc="训练中"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gt_activations = batch['ground_truth_activations'].to(device)

            # 学生模型预测的激活度
            pred_activations = student_model({'input_ids': input_ids, 'attention_mask': attention_mask})

            # 我们希望两个向量相似，所以目标是1
            target = torch.ones(pred_activations.size(0)).to(device)

            loss = loss_fn(pred_activations, gt_activations, target)
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"平均训练损失 (Cosine Embedding Loss): {avg_train_loss:.4f}")

        if val_dataloader:
            student_model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="验证中"):
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    gt_activations = batch['ground_truth_activations'].to(device)
                    pred_activations = student_model({'input_ids': input_ids, 'attention_mask': attention_mask})
                    target = torch.ones(pred_activations.size(0)).to(device)
                    loss = loss_fn(pred_activations, gt_activations, target)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"平均验证损失: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print("验证损失降低，保存模型 g ...")
                torch.save(student_model.state_dict(), MODEL_G_PATH)
        else:
            print("无验证集，保存当前epoch的模型 g ...")
            torch.save(student_model.state_dict(), MODEL_G_PATH)

    print("概念编码器训练完成。")
    student_model.load_state_dict(torch.load(MODEL_G_PATH))
    return student_model, student_tokenizer


def get_concept_activations(model_g, data, tokenizer):
    """为数据集生成概念激活向量"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_g.to(device).eval()
    activations = []
    with torch.no_grad():
        for item in tqdm(data, desc="生成激活向量"):
            encoding = tokenizer.encode_plus(item['text'], max_length=MAX_LENGTH, padding='max_length', truncation=True,
                                             return_tensors='pt').to(device)
            # 输出直接就是激活向量
            activation_vector = model_g(encoding).squeeze(0).cpu().numpy()
            activations.append(activation_vector)
    return np.array(activations)


def train_linear_predictor(model_g, train_data, val_data, tokenizer):
    """训练线性分类器 f"""
    print("\n--- 开始训练线性分类器 (f) ---")

    X_train = get_concept_activations(model_g, train_data, tokenizer)
    y_train_labels = [d['label'] for d in train_data]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)

    print("正在训练逻辑回归分类器...")
    model_f = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=42, C=0.1)
    model_f.fit(X_train, y_train)

    if val_data:
        print("在验证集上评估分类器...")
        X_val = get_concept_activations(model_g, val_data, tokenizer)
        y_val = label_encoder.transform([d['label'] for d in val_data])
        y_pred = model_f.predict(X_val)
        print(classification_report(y_val, y_pred, target_names=label_encoder.classes_, zero_division=0))

    print("分类器训练完成。")
    return model_f, label_encoder


def analyze_and_report(model_f, label_encoder, concept_dict):
    """分析模型权重并生成全局报告"""
    print(f"\n--- 开始分析模型并生成全局可解释性报告到 '{GLOBAL_REPORT_PATH}' ---")
    report = ["=" * 50, " CBM 全局可解释性分析报告", "=" * 50, "\n本报告展示了对于每个情感类别，哪些概念的贡献度最高。\n"]

    for class_idx, class_name in enumerate(label_encoder.classes_):
        report.append(f"\n### 预测类别为 '{class_name.upper()}' 时，贡献度最高的概念：\n")
        weights = model_f.coef_[class_idx]
        concept_weights = pd.DataFrame({'concept': concept_dict, 'weight': weights})
        top_concepts = concept_weights.sort_values(by='weight', ascending=False).head(10)

        for _, row in top_concepts.iterrows():
            report.append(f"- {row['concept']} (权重: {row['weight']:.4f})")

    final_report = "\n".join(report)
    print(final_report)
    with open(GLOBAL_REPORT_PATH, "w", encoding='utf-8') as f:
        f.write(final_report)
    print(f"\n全局分析报告已保存到 '{GLOBAL_REPORT_PATH}'")


def predict_and_analyze_test_set(model_g, model_f, test_data, tokenizer, concept_dict, label_encoder):
    """在测试集上进行预测并生成详细的样本级报告"""
    print(f"\n--- 在测试集上进行预测并生成详细报告到 '{TEST_PREDICTIONS_PATH}' ---")

    report_lines = []

    X_test = get_concept_activations(model_g, test_data, tokenizer)
    y_test_labels = [d['label'] for d in test_data]
    y_test_true = label_encoder.transform(y_test_labels)
    y_test_pred = model_f.predict(X_test)

    print("\n测试集整体性能报告:")
    print(classification_report(y_test_true, y_test_pred, target_names=label_encoder.classes_, zero_division=0))

    for i, item in enumerate(tqdm(test_data, desc="生成测试报告")):
        report_lines.append(f"样本原文: {item['text']}")
        report_lines.append(f"真实标签: {item['label']}")

        pred_label = label_encoder.inverse_transform([y_test_pred[i]])[0]
        report_lines.append(f"预测标签: {pred_label}")

        activations = X_test[i]
        top_k_indices = np.argsort(activations)[-TOP_K_CONCEPTS:][::-1]
        report_lines.append(f"Top {TOP_K_CONCEPTS} 激活概念:")
        for idx in top_k_indices:
            report_lines.append(f"  - {concept_dict[idx]} (激活度: {activations[idx]:.4f})")

        report_lines.append("-" * 60)

    with open(TEST_PREDICTIONS_PATH, "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"测试集预测报告已保存到 '{TEST_PREDICTIONS_PATH}'")


def main():
    """主执行函数"""
    train_data = load_data(TRAIN_FILE)
    val_data = load_data(VAL_FILE)
    test_data = load_data(TEST_FILE)

    if not train_data:
        print("错误：训练数据未找到，无法继续。")
        return

    with open(CONCEPTS_DICT_FILE, 'r', encoding='utf-8') as f:
        concept_dictionary = json.load(f)

    # 训练概念编码器 g (RobertaCBL)
    model_g, tokenizer = train_cbm(train_data, val_data, concept_dictionary)

    # 训练线性分类器 f
    model_f, label_encoder = train_linear_predictor(model_g, train_data, val_data, tokenizer)

    # 保存模型
    print("\n--- 保存训练好的模型 ---")
    joblib.dump(model_f, MODEL_F_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"模型 'f' 和标签编码器已保存到 '{MODEL_F_PATH}' 和 '{LABEL_ENCODER_PATH}'")

    # 分析并生成报告
    analyze_and_report(model_f, label_encoder, concept_dictionary)

    if test_data:
        predict_and_analyze_test_set(model_g, model_f, test_data, tokenizer, concept_dictionary, label_encoder)

    print("\n所有步骤完成！")


if __name__ == '__main__':
    for fname in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not os.path.exists(fname):
            print(f"警告: '{fname}' 不存在。请确保您已成功运行step2，或创建一个空的占位文件。")
            open(fname, 'w').close()
    main()
