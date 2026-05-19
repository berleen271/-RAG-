import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re, requests
from config import QWEN_API_KEY, QWEN_URL_TEXT
from fact_checker import batch_fact_check

def generate_answer(query, context):
    ctx_str = "\n".join([f"[{c['src']}|P{c['page']}] {c['content'][:300]}" for c in context])
    prompt = f"""基于证据回答。证据不足则说“未找到”。\n证据：\n{ctx_str}\n问题：{query}\n答案（标注引用页码，如[Page 3]）："""
    try:
        resp = requests.post(
            QWEN_URL_TEXT,
            headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
            json={"model": "qwen2.5-7b-instruct", "input": {"messages": [{"role": "user", "content": prompt}]}},
            timeout=20
        )
        resp.raise_for_status()
        return resp.json()["output"]["text"]
    except:
        return "生成出错"


def verify_and_clean(ans, context, embedder):
    sentences = re.split(r'(?<=[。！？.!?])\s*', ans)
    valid_sents = [s.strip() for s in sentences if len(s.strip()) > 5]
    supported = batch_fact_check(valid_sents, [c["content"] for c in context], embedder)
    verified = []
    hallucination = False
    for sent, flag in zip(valid_sents, supported):
        if not flag:
            hallucination = True
            verified.append(f"⚠️ {sent}")
        else:
            verified.append(sent)
    clean_ans = " ".join(verified)

    # 更新后的页码提取正则
    page_pattern = r"\[(?:text|visual|table|figure)?\s*\|?\s*[Pp](?:age)?\s*:?\s*(\d+)\]"
    cited_pages = set(map(int, re.findall(page_pattern, ans)))

    valid_pages = set(c["page"] for c in context)
    verified_pages = [p for p in cited_pages if p in valid_pages]

    if hallucination:
        clean_ans += "\n\n*(系统校验：部分语句未被证据支持)*"
    return clean_ans, verified_pages, context