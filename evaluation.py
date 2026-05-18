import re
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_retrieval_layer(true_pages, retrieved_pages, k=3):
    if not true_pages: return {"Recall@k":0.0,"Precision@k":0.0,"MRR":0.0}
    true_set = set(true_pages)
    retrieved_k = retrieved_pages[:k]
    recall = len(set(retrieved_k) & true_set) / len(true_set)
    precision = len(set(retrieved_k) & true_set) / len(retrieved_k) if retrieved_k else 0.0
    mrr = 0.0
    for rank, p in enumerate(retrieved_k, 1):
        if p in true_set:
            mrr = 1.0 / rank
            break
    return {"Recall@k":recall, "Precision@k":precision, "MRR":mrr}

def evaluate_chunk_layer(true_chunks, retrieved_chunks, k=5):
    if not true_chunks: return {"Recall@k":0.0,"Precision@k":0.0,"MRR":0.0}
    retrieved_k = retrieved_chunks[:k]
    recall = len(set(retrieved_k) & set(true_chunks)) / len(set(true_chunks))
    precision = len(set(retrieved_k) & set(true_chunks)) / len(retrieved_k) if retrieved_k else 0.0
    mrr = 0.0
    for rank, c in enumerate(retrieved_k, 1):
        if c in true_chunks:
            mrr = 1.0 / rank
            break
    return {"Recall@k":recall, "Precision@k":precision, "MRR":mrr}

def evaluate_answer(pred, true):
    em = 1.0 if true[:20] in pred else 0.0
    pred_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", pred.lower()))
    true_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", true.lower()))
    if len(pred_words) + len(true_words) == 0:
        f1 = 0.0
    else:
        common = pred_words & true_words
        prec = len(common) / len(pred_words) if pred_words else 0.0
        rec = len(common) / len(true_words) if true_words else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    bleu_1 = 0.0
    rouge_l = 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu
        bleu_1 = sentence_bleu([true.split()], pred.split(), weights=(1,0,0,0))
    except ImportError: pass
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = scorer.score(true, pred)['rougeL'].fmeasure
    except ImportError: pass
    return {"EM":em, "F1":f1, "BLEU-1":bleu_1, "ROUGE-L":rouge_l}

def evaluate_evidence_layer(sentences, evidences, embedder, fact_checker):
    if not sentences: return 0.0
    supported = fact_checker(sentences, evidences, embedder)
    return sum(supported) / len(sentences)