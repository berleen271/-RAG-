import streamlit as st
import fitz
import numpy as np
import chromadb
import requests
import re
import os
import tempfile
import hashlib
import json
import warnings
import ast
from PIL import Image
import io
import base64
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import pytesseract
import shutil
import sys
import subprocess
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

# ===================== 安全与配置 =====================
QWEN_API_KEY = os.getenv("QWEN_API_KEY", st.secrets.get("QWEN_API_KEY", "sk-YOUR-KEY-HERE"))
if "YOUR-KEY" in QWEN_API_KEY:
    st.sidebar.warning("⚠️ 请配置 QWEN_API_KEY")

QWEN_URL_TEXT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
QWEN_URL_VL   = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# ===================== 跨平台 OCR 路径 =====================
def find_tesseract():
    env_path = os.getenv("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        return env_path
    if sys.platform.startswith("win"):
        candidates = [
            r"D:\新建文件夹\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]
    else:
        candidates = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

tesseract_path = find_tesseract()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# ===================== 模型加载 =====================
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from transformers import CLIPProcessor, CLIPModel
    import torch
except ImportError:
    st.error("请安装: pip install sentence-transformers torch torchvision transformers")
    st.stop()

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.to(device)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512, device=device)
    return st_model, clip_model, clip_proc, reranker, device

st_model, clip_model, clip_proc, reranker, device = load_models()

# ===================== 嵌入器 =====================
class UnifiedEmbedder:
    def __init__(self, st_model, clip_model, clip_proc, device):
        self.st_model = st_model
        self.clip_model = clip_model
        self.clip_proc = clip_proc
        self.device = device
        self.st_dim = 384
        self.clip_dim = 768

    def encode_text_st(self, texts):
        if isinstance(texts, str): texts = [texts]
        return self.st_model.encode(texts).tolist()

    def encode_text_clip(self, texts):
        if isinstance(texts, str): texts = [texts]
        inputs = self.clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)
            feats = outputs.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()

    def encode_image_clip(self, pil_images):
        if not pil_images: return []
        inputs = self.clip_proc(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
            feats = outputs.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()

embedder = UnifiedEmbedder(st_model, clip_model, clip_proc, device)

# ===================== 临时图片管理 =====================
TEMP_IMG_DIR = None

def init_temp_dir():
    global TEMP_IMG_DIR
    if TEMP_IMG_DIR and os.path.exists(TEMP_IMG_DIR):
        shutil.rmtree(TEMP_IMG_DIR, ignore_errors=True)
    TEMP_IMG_DIR = tempfile.mkdtemp(prefix="rag_imgs_")

def save_image_get_path(pil_img, page_num, element_idx):
    global TEMP_IMG_DIR
    if TEMP_IMG_DIR is None:
        TEMP_IMG_DIR = tempfile.mkdtemp(prefix="rag_imgs_")
    filename = f"p{page_num}_{element_idx}.png"
    path = os.path.join(TEMP_IMG_DIR, filename)
    pil_img.save(path)
    return filename

def load_image_from_path(img_path):
    global TEMP_IMG_DIR
    if not img_path or TEMP_IMG_DIR is None:
        return None
    full = os.path.join(TEMP_IMG_DIR, img_path)
    if os.path.exists(full):
        return Image.open(full)
    return None

# ===================== 工具函数 =====================
def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def b64_to_pil(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

def get_file_hash(b):
    return hashlib.md5(b).hexdigest()

# ===================== 语义分块 =====================
def detect_heading(text, font_size, is_bold):
    if re.match(r"^(第[一二三四五六七八九十\d]+章|[IVXLCDM]+\.|\d+(\.\d+)*)\s", text):
        return True
    if font_size > 13 and is_bold:
        return True
    return False

def semantic_chunk_page(page):
    text_dict = page.get_text("dict")
    blocks = text_dict["blocks"]
    blocks_sorted = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]) if "bbox" in b else (0,0))
    chunks = []
    current = {"type": "text", "content": "", "bbox": None}

    for block in blocks_sorted:
        if block["type"] == 0:
            for line in block["lines"]:
                line_text = ""
                line_bbox = None
                avg_font = 0
                is_bold = False
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text: continue
                    line_text += text + " "
                    avg_font += span["size"]
                    if "Bold" in span["font"]: is_bold = True
                    if line_bbox is None:
                        line_bbox = list(span["bbox"])
                    else:
                        line_bbox[0] = min(line_bbox[0], span["bbox"][0])
                        line_bbox[1] = min(line_bbox[1], span["bbox"][1])
                        line_bbox[2] = max(line_bbox[2], span["bbox"][2])
                        line_bbox[3] = max(line_bbox[3], span["bbox"][3])
                line_text = line_text.strip()
                if not line_text: continue
                avg_font = avg_font / len(line["spans"]) if line["spans"] else 10

                if detect_heading(line_text, avg_font, is_bold):
                    if current["content"].strip():
                        chunks.append(current)
                    current = {"type": "heading", "content": line_text, "bbox": line_bbox}
                else:
                    if current["bbox"] is None:
                        current["bbox"] = line_bbox
                    else:
                        if line_bbox:
                            current["bbox"][0] = min(current["bbox"][0], line_bbox[0])
                            current["bbox"][1] = min(current["bbox"][1], line_bbox[1])
                            current["bbox"][2] = max(current["bbox"][2], line_bbox[2])
                            current["bbox"][3] = max(current["bbox"][3], line_bbox[3])
                    if current["type"] == "heading":
                        current["content"] += " " + line_text
                        current["type"] = "text"
                    else:
                        current["content"] += " " + line_text
        elif block["type"] == 1:
            if current["content"].strip():
                chunks.append(current)
            current = {"type": "text", "content": "", "bbox": None}
            chunks.append({"type": "figure", "content": "", "bbox": block["bbox"]})

    if current["content"].strip():
        chunks.append(current)
    return chunks

# ===================== 版面提取 =====================
def extract_layout_structured(page, page_num, zoom=2):
    elements = []
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    words = page.get_text("words")
    if words:
        xs = [w[0] for w in words] + [w[2] for w in words]
        ys = [w[1] for w in words] + [w[3] for w in words]
        text_rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
        coverage = (text_rect.width * text_rect.height) / (page.rect.width * page.rect.height)
    else:
        coverage = 0.0

    ocr_text = ""
    if coverage < 0.1 and tesseract_path:
        try:
            ocr_text = pytesseract.image_to_string(page_img, lang="chi_sim+eng")
        except:
            ocr_text = ""

    chunks = semantic_chunk_page(page)
    if not chunks and ocr_text.strip():
        chunks = [{"type": "text", "content": ocr_text, "bbox": (0,0,page.rect.width*zoom,page.rect.height*zoom)}]

    for chunk in chunks:
        if chunk["type"] in ["heading", "text"] and len(chunk["content"]) > 10:
            elements.append({
                "type": chunk["type"], "page": page_num,
                "content": chunk["content"],
                "bbox": chunk["bbox"] if chunk["bbox"] else (0,0,0,0),
                "img_path": None
            })

    tab_finder = page.find_tables()
    if tab_finder:
        for idx, t in enumerate(tab_finder.tables):
            data = t.extract()
            if not data or len(data) < 2: continue
            md = "\n".join(["| " + " | ".join([str(c).strip() if c else "" for c in r]) + " |" for r in data])
            tab_bbox = t.bbox
            crop = page_img.crop((tab_bbox[0]*zoom, tab_bbox[1]*zoom, tab_bbox[2]*zoom, tab_bbox[3]*zoom))
            if crop is None or crop.width <= 0 or crop.height <= 0:
                continue
            img_path = save_image_get_path(crop, page_num, len(elements))
            elements.append({
                "type": "table", "page": page_num,
                "content": f"[TABLE]\n{md}",
                "bbox": tab_bbox, "img_path": img_path
            })

    for idx, img_info in enumerate(page.get_images()):
        bbox = page.get_image_bbox(img_info)
        if bbox is None: continue
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        area_ratio = (w*h) / (page.rect.width * page.rect.height)
        if area_ratio < 0.05 or bbox[1] < page.rect.height*0.1 or bbox[3] > page.rect.height*0.9:
            continue
        clip_bbox = (bbox[0]-20, bbox[3]+5, bbox[2]+20, bbox[3]+40)
        caption = page.get_text("text", clip=clip_bbox).strip()
        is_figure = bool(re.search(r"(图|Fig|Figure|Chart)\s*\d+", caption, re.I))
        if not (is_figure or area_ratio > 0.1): continue

        crop = page_img.crop((bbox[0]*zoom, bbox[1]*zoom, bbox[2]*zoom, bbox[3]*zoom))
        if crop is None or crop.width <= 0 or crop.height <= 0:
            continue

        chart_types = ["chart", "table", "photo", "logo", "formula"]
        text_inputs = clip_proc(text=chart_types, return_tensors="pt", padding=True).to(device)
        img_inputs = clip_proc(images=crop, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = clip_model(**img_inputs, **text_inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        pred_type = chart_types[np.argmax(probs)]

        content = "[IMAGE]"
        img_path = save_image_get_path(crop, page_num, len(elements))

        if pred_type in ["chart", "table"]:
            b64 = pil_to_b64(crop)
            json_text = None
            for attempt in range(2):
                try:
                    resp = requests.post(
                        QWEN_URL_VL,
                        headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": "qwen-vl-plus",
                            "input": {
                                "messages": [{
                                    "role": "user",
                                    "content": [
                                        {"image": f"data:image/png;base64,{b64}"},
                                        {"text": "提取图表数据，仅输出纯JSON：{\"x\":[],\"y\":[],\"trend\":\"\"}，不要Markdown。"}
                                    ]
                                }]
                            }
                        },
                        timeout=8
                    )
                    raw = resp.json()["output"]["choices"][0]["message"]["content"]
                    raw = re.sub(r"```json|```", "", raw).strip()
                    parsed = json.loads(raw)
                    if all(k in parsed for k in ["x", "y", "trend"]):
                        json_text = json.dumps(parsed, ensure_ascii=False)
                        break
                except:
                    pass
            if json_text:
                content = f"[CHART_STRUCTURED]\n{json_text}"
            else:
                content = "[CHART]（未能解析）"
                st.toast("⚠️ 部分图表未能提取结构化数据", icon="⚠️")

        elements.append({
            "type": "figure", "page": page_num,
            "content": content,
            "bbox": bbox, "img_path": img_path
        })
    return elements

# ===================== 双通道页向量 =====================
def build_page_vectors(page_items):
    st_vecs, clip_vecs = [], []
    for item in page_items:
        if item["type"] in ["heading", "text"] and item["content"]:
            st_vecs.append(embedder.encode_text_st([item["content"]])[0])
            clip_vecs.append(embedder.encode_text_clip([item["content"]])[0])
        elif item["type"] in ["table", "figure"] and item.get("img_path"):
            img = load_image_from_path(item["img_path"])
            if img:
                clip_vecs.append(embedder.encode_image_clip([img])[0])
            if not item["content"].startswith("[TABLE]") and not item["content"].startswith("[CHART"):
                st_vecs.append(embedder.encode_text_st([item["content"]])[0])
    p_st = np.mean(st_vecs, axis=0).tolist() if st_vecs else [0.0]*embedder.st_dim
    p_clip = np.mean(clip_vecs, axis=0).tolist() if clip_vecs else [0.0]*embedder.clip_dim
    return p_st, p_clip

# ===================== 索引构建 =====================
@st.cache_resource
def get_chroma():
    return chromadb.PersistentClient("./chroma_v12")

def build_index(all_data):
    init_temp_dir()
    client = get_chroma()
    for name in ["page_text_idx", "page_visual_idx", "text_idx", "visual_idx"]:
        try: client.delete_collection(name)
        except: pass
    p_tc = client.get_or_create_collection("page_text_idx", embedding_function=None)
    p_vc = client.get_or_create_collection("page_visual_idx", embedding_function=None)
    t_c  = client.get_or_create_collection("text_idx", embedding_function=None)
    v_c  = client.get_or_create_collection("visual_idx", embedding_function=None)

    p_t_ids, p_t_vecs, p_t_meta = [], [], []
    p_v_ids, p_v_vecs, p_v_meta = [], [], []
    t_ids, t_vecs, t_docs, t_meta = [], [], [], []
    v_ids, v_vecs, v_docs, v_meta = [], [], [], []

    for pnum, items in all_data.items():
        p_st, p_clip = build_page_vectors(items)
        p_t_ids.append(f"p_{pnum}"); p_t_vecs.append(p_st); p_t_meta.append({"page":pnum})
        p_v_ids.append(f"p_{pnum}"); p_v_vecs.append(p_clip); p_v_meta.append({"page":pnum})

        for idx, it in enumerate(items):
            meta = {
                "page": pnum, "type": it["type"],
                "bbox": str(it["bbox"]),
                "img_path": it.get("img_path") or ""
            }
            if it["content"]:
                vec = embedder.encode_text_st([it["content"]])[0]
                t_ids.append(f"t_{len(t_ids)}"); t_vecs.append(vec)
                t_docs.append(it["content"]); t_meta.append(meta)
            if it["type"] in ["table", "figure"] and it.get("img_path"):
                img = load_image_from_path(it["img_path"])
                if img:
                    v_vec = embedder.encode_image_clip([img])[0]
                    v_ids.append(f"v_{len(v_ids)}"); v_vecs.append(v_vec)
                    v_docs.append(it["content"]); v_meta.append(meta)

    if p_t_ids: p_tc.add(ids=p_t_ids, embeddings=p_t_vecs, metadatas=p_t_meta)
    if p_v_ids: p_vc.add(ids=p_v_ids, embeddings=p_v_vecs, metadatas=p_v_meta)
    if t_ids: t_c.add(ids=t_ids, embeddings=t_vecs, documents=t_docs, metadatas=t_meta)
    if v_ids: v_c.add(ids=v_ids, embeddings=v_vecs, documents=v_docs, metadatas=v_meta)
    return p_tc, p_vc, t_c, v_c

# ===================== 分层检索（支持 trace） =====================
def retrieve_hierarchical(query, p_tc, p_vc, t_c, v_c,
                          use_page=True, use_vis=True, dynamic_weight=True, top_k=20,
                          return_trace=False):
    q_st = embedder.encode_text_st([query])[0]
    q_clip = embedder.encode_text_clip([query])[0]

    cand_pages = set()
    if use_page:
        p_t_res = p_tc.query(query_embeddings=[q_st], n_results=3)
        cand_pages.update([m["page"] for m in p_t_res["metadatas"][0]])
        if use_vis:
            p_v_res = p_vc.query(query_embeddings=[q_clip], n_results=3)
            cand_pages.update([m["page"] for m in p_v_res["metadatas"][0]])
    else:
        t_all = t_c.query(query_embeddings=[q_st], n_results=top_k*2)
        v_all = v_c.query(query_embeddings=[q_clip], n_results=top_k)
        cand_pages = set([m["page"] for m in t_all["metadatas"][0]] +
                         [m["page"] for m in v_all["metadatas"][0]])
    if not cand_pages:
        if return_trace:
            return [], [], [], []
        return []

    t_res = t_c.query(query_embeddings=[q_st], n_results=top_k, where={"page":{"$in":list(cand_pages)}})
    v_res = v_c.query(query_embeddings=[q_clip], n_results=top_k//2, where={"page":{"$in":list(cand_pages)}})

    w_txt, w_vis = 0.6, 0.4
    if dynamic_weight:
        ql = query.lower()
        if any(k in ql for k in ["表","table","数据","金额"]):
            w_txt, w_vis = 0.3, 0.7
        elif any(k in ql for k in ["图","chart","趋势","走势"]):
            w_txt, w_vis = 0.2, 0.8

    def dist2sim(dists):
        return [1.0/(1.0+d) for d in dists]

    txt_sims = dist2sim(t_res["distances"][0]) if t_res["ids"] else []
    vis_sims = dist2sim(v_res["distances"][0]) if v_res["ids"] else []

    merged = []
    for i, m in enumerate(t_res["metadatas"][0]):
        merged.append({
            "score": w_txt * txt_sims[i],
            "src": "text", "page": m["page"],
            "content": t_res["documents"][0][i],
            "bbox_str": m["bbox"], "img_path": m.get("img_path")
        })
    if use_vis:
        for i, m in enumerate(v_res["metadatas"][0]):
            merged.append({
                "score": w_vis * vis_sims[i],
                "src": "visual", "page": m["page"],
                "content": v_res["documents"][0][i],
                "bbox_str": m["bbox"], "img_path": m.get("img_path")
            })
    merged.sort(key=lambda x: x["score"], reverse=True)
    # 保存重排序前的Top-k（深拷贝）
    pre_rerank_dicts = [dict(item) for item in merged[:top_k]]

    if len(pre_rerank_dicts) > 1:
        pairs = [[query, c["content"][:400]] for c in pre_rerank_dicts[:15]]
        ce_scores = reranker.predict(pairs)
        for i, c in enumerate(pre_rerank_dicts[:15]):
            c["score"] = float(ce_scores[i])
        pre_rerank_dicts.sort(key=lambda x: x["score"], reverse=True)

    final_context = pre_rerank_dicts[:6]

    if return_trace:
        return final_context, list(cand_pages), merged[:top_k], final_context
    return final_context

# ===================== 事实核查 =====================
def batch_fact_check(sentences, evidences):
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
        try:
            arr = json.loads(raw)
            if isinstance(arr, list): return arr
        except:
            pass
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except:
        pass
    s_vecs = embedder.encode_text_st(sentences)
    ev_vecs = embedder.encode_text_st(evidences)
    return [float(np.max(cosine_similarity([v], ev_vecs))) > 0.7 for v in s_vecs]

# ===================== 生成与验证 =====================
def generate_and_verify(query, context):
    ctx_str = "\n".join([f"[{c['src']}|P{c['page']}] {c['content'][:300]}" for c in context])
    prompt = f"""基于证据回答。证据不足则说“未找到”。
证据：
{ctx_str}
问题：{query}
答案（标注引用页码，如[Page 3]）："""
    try:
        resp = requests.post(
            QWEN_URL_TEXT,
            headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
            json={"model": "qwen2.5-7b-instruct", "input": {"messages": [{"role": "user", "content": prompt}]}},
            timeout=20
        )
        resp.raise_for_status()
        ans = resp.json()["output"]["text"]
    except Exception as e:
        return "生成出错", [], context

    sentences = re.split(r'(?<=[。！？.!?])\s*', ans)
    valid_sents = [s.strip() for s in sentences if len(s.strip()) > 5]
    supported = batch_fact_check(valid_sents, [c["content"] for c in context])
    verified = []
    hallucination = False
    for sent, flag in zip(valid_sents, supported):
        if not flag:
            hallucination = True
            verified.append(f"⚠️ {sent}")
        else:
            verified.append(sent)
    clean_ans = " ".join(verified)
    page_pattern = r"\[(?:Page|P|page|p)\s*:?\s*(\d+)\]"
    cited_pages = set(map(int, re.findall(page_pattern, ans)))
    valid_pages = set(c["page"] for c in context)
    verified_pages = [p for p in cited_pages if p in valid_pages]
    if hallucination:
        clean_ans += "\n\n*(系统校验：部分语句未被证据支持)*"
    return clean_ans, verified_pages, context

# ===================== 评估函数 =====================
def evaluate_retrieval_layer(true_pages: List[int], retrieved_pages: List[int], k=3) -> Dict[str, float]:
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

def evaluate_chunk_layer(true_chunks: List[str], retrieved_chunks: List[str], k=5) -> Dict[str, float]:
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

def evaluate_answer(pred: str, true: str) -> Dict[str, float]:
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
    except ImportError:
        pass
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = scorer.score(true, pred)['rougeL'].fmeasure
    except ImportError:
        pass
    return {"EM": em, "F1": f1, "BLEU-1": bleu_1, "ROUGE-L": rouge_l}

def evaluate_evidence_layer(sentences: List[str], evidences: List[str]) -> float:
    if not sentences:
        return 0.0
    supported = batch_fact_check(sentences, evidences)
    return sum(supported) / len(sentences)

# ===================== 消融实验（三层评估） =====================
def run_full_ablation_study(qa_set, p_tc, p_vc, t_c, v_c):
    setups = [
        {"name": "Baseline (Flat Text)",      "use_page": False, "use_vis": False, "dynamic": False},
        {"name": "ExpA (+Hierarchical)",      "use_page": True,  "use_vis": False, "dynamic": False},
        {"name": "ExpB (+Visual)",            "use_page": True,  "use_vis": True,  "dynamic": False},
        {"name": "ExpC (+Dynamic Weight)",    "use_page": True,  "use_vis": True,  "dynamic": True},
    ]
    results = []
    for s in setups:
        page_recalls, page_precisions, page_mrrs = [], [], []
        chunk_recalls_pre, chunk_precisions_pre, chunk_mrrs_pre = [], [], []
        chunk_recalls_post, chunk_precisions_post, chunk_mrrs_post = [], [], []
        answer_em, answer_f1, bleu_scores, rouge_scores = [], [], [], []
        ssr_scores = []

        for q, true_ans, true_pages, true_chunks in qa_set:
            final_ctx, pages, pre_rerank_dicts, post_rerank_dicts = retrieve_hierarchical(
                q, p_tc, p_vc, t_c, v_c,
                use_page=s["use_page"], use_vis=s["use_vis"],
                dynamic_weight=s["dynamic"], return_trace=True
            )
            # 评估页面
            pr = evaluate_retrieval_layer(true_pages, pages, k=3)
            page_recalls.append(pr["Recall@k"])
            page_precisions.append(pr["Precision@k"])
            page_mrrs.append(pr["MRR"])

            # 评估 chunk pre-rerank
            pre_contents = [c["content"] for c in pre_rerank_dicts] if pre_rerank_dicts else []
            cr_pre = evaluate_chunk_layer(true_chunks, pre_contents, k=5)
            chunk_recalls_pre.append(cr_pre["Recall@k"])
            chunk_precisions_pre.append(cr_pre["Precision@k"])
            chunk_mrrs_pre.append(cr_pre["MRR"])

            # 评估 chunk post-rerank
            post_contents = [c["content"] for c in post_rerank_dicts] if post_rerank_dicts else []
            cr_post = evaluate_chunk_layer(true_chunks, post_contents, k=5)
            chunk_recalls_post.append(cr_post["Recall@k"])
            chunk_precisions_post.append(cr_post["Precision@k"])
            chunk_mrrs_post.append(cr_post["MRR"])

            # 答案生成与评估
            pred_ans, vp, ctx_full = generate_and_verify(q, final_ctx)
            ans_eval = evaluate_answer(pred_ans, true_ans)
            answer_em.append(ans_eval["EM"])
            answer_f1.append(ans_eval["F1"])
            bleu_scores.append(ans_eval["BLEU-1"])
            rouge_scores.append(ans_eval["ROUGE-L"])

            # 证据层 SSR
            sentences = re.split(r'(?<=[。！？.!?])\s*', pred_ans)
            valid_sents = [s.strip() for s in sentences if len(s.strip()) > 5]
            ssr = evaluate_evidence_layer(valid_sents, [c["content"] for c in ctx_full])
            ssr_scores.append(ssr)

        n = len(qa_set)
        results.append({
            "Setup": s["name"],
            "Page R@3": f"{np.mean(page_recalls):.3f}",
            "Page P@3": f"{np.mean(page_precisions):.3f}",
            "Page MRR": f"{np.mean(page_mrrs):.3f}",
            "Chunk pre R@5": f"{np.mean(chunk_recalls_pre):.3f}",
            "Chunk pre P@5": f"{np.mean(chunk_precisions_pre):.3f}",
            "Chunk pre MRR": f"{np.mean(chunk_mrrs_pre):.3f}",
            "Chunk post R@5": f"{np.mean(chunk_recalls_post):.3f}",
            "Chunk post P@5": f"{np.mean(chunk_precisions_post):.3f}",
            "Chunk post MRR": f"{np.mean(chunk_mrrs_post):.3f}",
            "Answer EM": f"{np.mean(answer_em):.3f}",
            "Answer F1": f"{np.mean(answer_f1):.3f}",
            "Answer BLEU-1": f"{np.mean(bleu_scores):.3f}",
            "Answer ROUGE-L": f"{np.mean(rouge_scores):.3f}",
            "SSR (Evidence)": f"{np.mean(ssr_scores):.3f}"
        })
    return pd.DataFrame(results)

# ===================== 测试集生成 =====================
def auto_generate_qa_with_chunks(all_data, num_text=15, num_table=10, num_figure=10):
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

# ===================== 端到端可用性评估（人工） =====================
def collect_human_feedback():
    if "feedback" not in st.session_state:
        st.session_state.feedback = []
    with st.sidebar:
        st.subheader("📝 端到端可用性评估")
        score = st.slider("满意度评分（1-5）", 1, 5, 3, key="satisfaction")
        safety = st.radio("回答是否安全/合规？", ["是", "否（包含错误或有害内容）"], key="safety")
        if st.button("提交评价"):
            if st.session_state.get("last_answer"):
                st.session_state.feedback.append({
                    "question": st.session_state.get("last_question", ""),
                    "answer": st.session_state.last_answer,
                    "score": score,
                    "safety": safety
                })
                st.success("评价已记录")
            else:
                st.warning("请先生成一次回答")

# ===================== Streamlit UI =====================
st.set_page_config(layout="wide", page_title="RAG 多模态问答（评估版）")
st.title("📚 多模态文档问答（含检索-答案-证据三层评估）")

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

col1, col2, col3 = st.columns([1, 1.8, 1.2])

with col1:
    st.header("📁 文档管理")
    up = st.file_uploader("上传 PDF", type="pdf")
    if st.button("🗑️ 清空会话", use_container_width=True):
        st.session_state.clear()
        if TEMP_IMG_DIR and os.path.exists(TEMP_IMG_DIR):
            shutil.rmtree(TEMP_IMG_DIR, ignore_errors=True)
        try:
            client = get_chroma()
            for name in ["page_text_idx", "page_visual_idx", "text_idx", "visual_idx"]:
                client.delete_collection(name)
        except: pass
        st.rerun()

    if up:
        hb = get_file_hash(up.read()); up.seek(0)
        if st.session_state.get("hash") != hb:
            with st.spinner("🧠 版面分析 + 双通道索引..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(up.read()); tmp.flush()
                    doc = fitz.open(tmp.name)
                    data = {}
                    for i, p in enumerate(doc):
                        data[i+1] = extract_layout_structured(p, i+1)
                    doc.close()
                    build_index(data)
                    st.session_state.all_data = data
                    st.session_state.pdf_path = tmp.name
                    st.session_state.hash = hb
            st.success("✅ 索引就绪")

    st.divider()
    st.subheader("📊 消融实验（三层评估）")
    if st.button("🚀 运行完整消融实验", use_container_width=True):
        if st.session_state.get("hash"):
            p_tc = get_chroma().get_collection("page_text_idx")
            p_vc = get_chroma().get_collection("page_visual_idx")
            t_c  = get_chroma().get_collection("text_idx")
            v_c  = get_chroma().get_collection("visual_idx")
            if "all_data" in st.session_state and len(st.session_state.all_data) > 0:
                qa = auto_generate_qa_with_chunks(st.session_state.all_data, num_text=5, num_table=3, num_figure=2)
            else:
                qa = [("表格中2023年营收是多少？", "2023年营收为500亿元。", [3], ["营收500亿"])]
            st.info(f"📋 测试集规模：{len(qa)} 条（自动生成，建议人工校验）")
            with st.spinner("🧪 正在运行消融实验（含检索、答案、证据三层评估）..."):
                res_df = run_full_ablation_study(qa, p_tc, p_vc, t_c, v_c)
            st.dataframe(res_df, use_container_width=True)
            st.success("实验完成，可在下方查看端到端反馈收集")
        else:
            st.warning("请先上传PDF")

    st.divider()
    st.subheader("📈 端到端可用性评价")
    collect_human_feedback()

with col2:
    st.header("💬 智能问答")
    for m in st.session_state.get("hist", []):
        st.chat_message(m["role"]).write(m["content"])
    if q := st.chat_input("请输入问题..."):
        if not st.session_state.get("hash"):
            st.error("请先上传PDF"); st.stop()
        st.chat_message("user").write(q)
        st.session_state.last_question = q
        with st.spinner("🔍 双通道检索 + 重排序 + 事实核查..."):
            p_tc = get_chroma().get_collection("page_text_idx")
            p_vc = get_chroma().get_collection("page_visual_idx")
            t_c  = get_chroma().get_collection("text_idx")
            v_c  = get_chroma().get_collection("visual_idx")
            ctx = retrieve_hierarchical(q, p_tc, p_vc, t_c, v_c)   # 不需要trace，只取最终上下文
            ans, vp, ctx_full = generate_and_verify(q, ctx)
            st.session_state.last_answer = ans
            hist = st.session_state.get("hist", [])
            hist.extend([{"role":"user","content":q}, {"role":"assistant","content":ans,"ctx":ctx_full,"vp":vp}])
            st.session_state.hist = hist
            st.rerun()

with col3:
    st.header("🔍 证据溯源")
    if st.session_state.get("hist") and st.session_state.hist[-1]["role"]=="assistant":
        last = st.session_state.hist[-1]
        st.success(f"✅ 已验证页码: {last['vp']}")
        for c in last["ctx"]:
            if c["page"] in last["vp"]:
                st.subheader(f"📄 第 {c['page']} 页 | {c['src']} | 相似度 {c['score']:.2f}")
                doc = fitz.open(st.session_state.pdf_path)
                pg = doc[c["page"]-1]
                pix = pg.get_pixmap(matrix=fitz.Matrix(2,2))
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img = img.copy()
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                try:
                    bbox = ast.literal_eval(c.get("bbox_str", "(0,0,0,0)"))
                except:
                    bbox = (0,0,0,0)
                x0,y0,x1,y1 = bbox
                if x0!=0 or y0!=0 or x1!=0 or y1!=0:
                    z = 2
                    pt1 = (int(x0*z), int(y0*z))
                    pt2 = (int(x1*z), int(y1*z))
                    cv2.rectangle(img, pt1, pt2, (0,255,0), 4)
                st.image(img)
                with st.expander("📜 证据原文", expanded=False):
                    st.text(c["content"][:400])
                doc.close()
            st.divider()
