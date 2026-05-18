import re, json, requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import QWEN_API_KEY, QWEN_URL_TEXT

def batch_fact_check(sentences, evidences, embedder):
    if not evidences:
        return [False] * len(sentences)
    ctx = "\n".join([f"证据{i}: {e[:200]}" for i, e in enumerate(evidences)])
    sent_str = "\n".join([f"{i}. {s}" for i, s in enumerate(sentences)])
    prompt = f"""基于证据判断以下每个陈述是否真实。仅输出JSON数组（true/false）。
证据：
{ctx}
陈述：
{sent_str}
JSON数组："""
    try:
        resp = requests.post(
            QWEN_URL_TEXT,
            headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
            json={"model": "qwen2.5-7b-instruct", "input": {"messages": [{"role": "user", "content": prompt}]}},
            timeout=10
        )
        raw = resp.json()["output"]["text"].strip()
        def safe_bool_list(lst):
            if not isinstance(lst, list): return None
            result = []
            for item in lst:
                if isinstance(item, bool): result.append(item)
                elif isinstance(item, str): result.append(item.lower() == "true")
                else: result.append(bool(item))
            return result
        try:
            arr = json.loads(raw)
            bool_list = safe_bool_list(arr)
            if bool_list is not None and len(bool_list) == len(sentences):
                return bool_list
        except: pass
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                arr = json.loads(m.group())
                bool_list = safe_bool_list(arr)
                if bool_list is not None and len(bool_list) == len(sentences):
                    return bool_list
            except: pass
    except: pass
    # fallback
    s_vecs = embedder.encode_text_st(sentences)
    ev_vecs = embedder.encode_text_st(evidences)
    return [float(np.max(cosine_similarity([v], ev_vecs))) > 0.7 for v in s_vecs]