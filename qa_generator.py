# qa_generator.py
import json
import re
import requests
from config import QWEN_API_KEY, QWEN_URL_TEXT


def generate_qa_for_chunk(chunk_text, chunk_type, page_num):
    """为一个证据块生成改写后的问题和答案"""
    prompt = f"""你是一个测试集构建助手。请根据以下文档片段，生成1条问答对。
要求：
1. 问题要用自然语言表达，不能直接复制原文的句子，必须改写。
2. 答案要简洁准确，基于原文回答，但不要照抄原文。
3. 输出JSON格式：{{"question":"...", "answer":"..."}}

文档片段（{chunk_type}，来自第{page_num}页）：
{chunk_text}
"""
    try:
        resp = requests.post(
            QWEN_URL_TEXT,
            headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
            json={"model": "qwen2.5-7b-instruct", "input": {"messages": [{"role": "user", "content": prompt}]}},
            timeout=10
        )
        resp.raise_for_status()
        result = resp.json()["output"]["text"].strip()
        m = re.search(r'\{.*?\}', result, re.DOTALL)
        if m:
            return json.loads(m.group())
    except:
        pass
    return None


def smart_auto_generate_qa(all_data, max_per_type=30):
    """智能生成测试集：调用大模型改写问题，每种类型最多生成 max_per_type 条"""
    qa_list = []
    text_count, table_count, figure_count = 0, 0, 0
    for pnum, items in all_data.items():
        for it in items:
            if it["type"] in ["heading", "text"] and text_count < max_per_type:
                qa = generate_qa_for_chunk(it["content"][:300], "text", pnum)
                if qa:
                    qa_list.append((qa["question"], qa["answer"], [pnum], [it["content"]]))
                    text_count += 1
            elif it["type"] == "table" and table_count < max_per_type:
                qa = generate_qa_for_chunk(it["content"][:200], "table", pnum)
                if qa:
                    qa_list.append((qa["question"], qa["answer"], [pnum], [it["content"]]))
                    table_count += 1
            elif it["type"] == "figure" and figure_count < max_per_type:
                qa = generate_qa_for_chunk(it["content"][:200], "figure", pnum)
                if qa:
                    qa_list.append((qa["question"], qa["answer"], [pnum], [it["content"]]))
                    figure_count += 1
    return qa_list


def auto_generate_qa_with_chunks(all_data, num_text=15, num_table=10, num_figure=10):
    """简单自动生成（回退方案）：直接从元素内容构造问题，结果可能虚高，仅用于演示"""
    qa_list = []
    for pnum, items in all_data.items():
        for it in items:
            if it["type"] in ["heading", "text"] and len(qa_list) < num_text:
                q = f"请解释第{pnum}页中关于'{it['content'][:30]}'的内容"
                ans = it["content"][:100]
                qa_list.append((q, ans, [pnum], [it["content"]]))
            elif it["type"] == "table" and len(qa_list) < num_text + num_table:
                lines = it["content"].split("\n")
                if len(lines) >= 2:
                    cells = [c.strip() for c in lines[1].split("|") if c.strip()]
                    if len(cells) >= 2:
                        q = f"在第{pnum}页的表格中，{cells[0]} 对应的数值是多少？"
                        ans = cells[1]
                    else:
                        q = f"表格 {pnum} 显示了什么？"
                        ans = lines[1][:100]
                else:
                    q = f"第{pnum}页的表格内容是什么？"
                    ans = it["content"][:100]
                qa_list.append((q, ans, [pnum], [it["content"]]))
            elif it["type"] == "figure" and len(qa_list) < num_text + num_table + num_figure:
                caption = it["content"].replace("[IMAGE_CAPTION]\n", "").replace("[CHART_STRUCTURED]\n", "")[:100]
                q = f"描述第{pnum}页的图像内容。"
                ans = caption
                qa_list.append((q, ans, [pnum], [it["content"]]))
    return qa_list