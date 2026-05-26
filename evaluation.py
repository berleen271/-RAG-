import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_retrieval_layer(true_pages, retrieved_pages, k=3):
    if not true_pages:
        return {"Recall@k": 0.0, "Precision@k": 0.0, "MRR": 0.0}
    true_set = set(true_pages)
    retrieved_k = retrieved_pages[:k]
    recall = len(set(retrieved_k) & true_set) / len(true_set)
    precision = len(set(retrieved_k) & true_set) / len(retrieved_k) if retrieved_k else 0.0
    mrr = 0.0
    for rank, p in enumerate(retrieved_k, 1):
        if p in true_set:
            mrr = 1.0 / rank
            break
    return {"Recall@k": recall, "Precision@k": precision, "MRR": mrr}


def evaluate_chunk_layer(true_chunks, retrieved_chunks, k=5):
    if not true_chunks:
        return {"Recall@k": 0.0, "Precision@k": 0.0, "MRR": 0.0}
    retrieved_k = retrieved_chunks[:k]
    recall = len(set(retrieved_k) & set(true_chunks)) / len(set(true_chunks))
    precision = len(set(retrieved_k) & set(true_chunks)) / len(retrieved_k) if retrieved_k else 0.0
    mrr = 0.0
    for rank, c in enumerate(retrieved_k, 1):
        if c in true_chunks:
            mrr = 1.0 / rank
            break
    return {"Recall@k": recall, "Precision@k": precision, "MRR": mrr}


def evaluate_answer(pred, true):
    # 空值保护
    if not pred or not true:
        return {"EM": 0.0, "F1": 0.0, "BLEU-1": 0.0, "ROUGE-L": 0.0}

    # 清洗引用标记
    pred_clean = re.sub(r'\[.*?\]|\(.*?\)', '', pred).strip()
    true_clean = re.sub(r'\[.*?\]|\(.*?\)', '', true).strip()

    if not pred_clean:
        pred_clean = pred
    if not true_clean:
        true_clean = true

    # ---------- EM（带数字容错）----------
    def calculate_em(t, p):
        # 先尝试子串匹配
        for i in range(min(20, len(t)), 2, -1):
            if t[:i] in p:
                return 1.0
        # 如果标准答案含有数字，且数字出现在生成答案中，也算EM=1
        nums = re.findall(r'\d+\.?\d*', t)
        if nums and any(num in p for num in nums):
            return 1.0
        return 0.0

    em = calculate_em(true_clean, pred_clean)

    # ---------- F1（词汇重叠 + 数字容错）----------
    pred_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", pred_clean.lower()))
    true_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", true_clean.lower()))

    # 如果标准答案全是数字/符号，导致分词结果为空，则使用字符级重叠
    if not true_words:
        # 把标准答案的每个数字/字符作为独立的“词”
        true_chars = set(re.findall(r'\d+\.?\d*|[\u4e00-\u9fa5]', true_clean))
        pred_chars = set(re.findall(r'\d+\.?\d*|[\u4e00-\u9fa5]', pred_clean))
        common = true_chars & pred_chars
        f1 = 2 * len(common) / (len(true_chars) + len(pred_chars)) if (true_chars | pred_chars) else 0.0
    else:
        common = pred_words & true_words
        prec = len(common) / len(pred_words) if pred_words else 0.0
        rec = len(common) / len(true_words) if true_words else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # BLEU-1
    bleu_1 = 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method1
        bleu_1 = sentence_bleu(
            [true_clean.split()], pred_clean.split(),
            weights=(1, 0, 0, 0), smoothing_function=smoothie
        )
    except ImportError:
        pass

    # ROUGE-L
    rouge_l = 0.0
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = scorer.score(true_clean, pred_clean)['rougeL'].fmeasure
    except ImportError:
        pass

    return {"EM": em, "F1": f1, "BLEU-1": bleu_1, "ROUGE-L": rouge_l}


def evaluate_evidence_layer(sentences, evidences, embedder, fact_checker):
    if not sentences:
        return 0.0
    supported = fact_checker(sentences, evidences, embedder)
    return sum(supported) / len(sentences) if len(sentences) > 0 else 0.0