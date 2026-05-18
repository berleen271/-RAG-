import streamlit as st
import fitz
import tempfile
import ast
import cv2
import numpy as np
import os, json, shutil, sys
from pathlib import Path

# 本地模块
from config import *
from ocr_utils import setup_ocr
from embedder import load_models, UnifiedEmbedder
from utils import TempImageManager, get_file_hash
from layout_parser import extract_layout_structured
from indexer import get_chroma, build_index
from retriever import retrieve_hierarchical
from generator import generate_answer, verify_and_clean
from ablation import run_ablation_study

# ---- 初始设置 ----
tesseract_ok = setup_ocr()
st_model, clip_model, clip_proc, reranker, device = load_models()
embedder = UnifiedEmbedder(st_model, clip_model, clip_proc, device)
img_manager = TempImageManager()

if "all_data" not in st.session_state:
    st.session_state.all_data = {}
if "hash" not in st.session_state:
    st.session_state.hash = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "hist" not in st.session_state:
    st.session_state.hist = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# ---- UI ----
st.set_page_config(layout="wide", page_title="多模态RAG评估系统")
st.title("📚 多模态文档问答（模块化 + 人工标注测试集）")

col1, col2, col3 = st.columns([1, 1.8, 1.2])

with col1:
    st.header("📁 文档管理")
    up = st.file_uploader("上传 PDF", type="pdf")
    if st.button("🗑️ 清空会话", use_container_width=True):
        st.session_state.clear()
        img_manager.cleanup()
        try:
            client = get_chroma()
            for name in ["page_text_idx","page_visual_idx","text_idx","visual_idx"]:
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
                        data[i+1] = extract_layout_structured(p, i+1, zoom=2, img_manager=img_manager,
                                                               clip_proc=clip_proc, clip_model=clip_model,
                                                               device=device, tesseract_available=tesseract_ok)
                    doc.close()
                    img_manager.init_dir()  # 保证有新目录
                    build_index(data, embedder, img_manager)
                    st.session_state.all_data = data
                    st.session_state.pdf_path = tmp.name
                    st.session_state.hash = hb
            st.success("✅ 索引就绪")

    st.divider()
    st.subheader("📊 消融实验（真实人工标注测试集）")
    if st.button("🚀 运行消融实验", use_container_width=True):
        if st.session_state.get("hash"):
            p_tc = get_chroma().get_collection("page_text_idx")
            p_vc = get_chroma().get_collection("page_visual_idx")
            t_c = get_chroma().get_collection("text_idx")
            v_c = get_chroma().get_collection("visual_idx")

            # 尝试加载人工标注测试集
            qa_list = None
            if os.path.exists("data/test_qa.json"):
                with open("data/test_qa.json", "r", encoding="utf-8") as f:
                    qa_data = json.load(f)
                qa_list = [(q["question"], q["answer"], q["pages"], q["evidence_chunks"]) for q in qa_data]
                st.success("已加载人工标注测试集")
            else:
                # 回退：自动生成少量 QA（仅用于演示，结果不可靠）
                from qa_generator import auto_generate_qa_with_chunks

                qa_list = auto_generate_qa_with_chunks(st.session_state.all_data, num_text=3, num_table=2, num_figure=1)
                st.warning("⚠️ 未找到 data/test_qa.json，使用自动生成的 QA（指标虚高，仅作演示）")

            st.info(f"📋 测试集规模：{len(qa_list)} 条")
            with st.spinner("🧪 运行消融实验..."):
                res_df = run_ablation_study(qa_list, p_tc, p_vc, t_c, v_c, embedder, reranker)
            st.dataframe(res_df, use_container_width=True)
        else:
            st.warning("请先上传PDF")

    st.divider()
    st.subheader("📈 端到端可用性评价")
    with st.sidebar:
        st.subheader("人工评价")
        score = st.slider("满意度（1-5）",1,5,3)
        safety = st.radio("是否安全合规？",["是","否"])
        if st.button("提交评价"):
            if st.session_state.last_answer:
                st.session_state.feedback.append({
                    "question": st.session_state.last_question,
                    "answer": st.session_state.last_answer,
                    "score": score,
                    "safety": safety
                })
                st.success("已记录")
            else:
                st.warning("请先生成回答")

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
            ctx = retrieve_hierarchical(q, p_tc, p_vc, t_c, v_c, embedder, reranker, use_page=True, use_vis=True, dynamic_weight=True)
            ans = generate_answer(q, ctx)
            clean_ans, vp, ctx_full = verify_and_clean(ans, ctx, embedder)
            st.session_state.last_answer = clean_ans
            hist = st.session_state.get("hist", [])
            hist.extend([{"role":"user","content":q}, {"role":"assistant","content":clean_ans,"ctx":ctx_full,"vp":vp}])
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
                if st.session_state.pdf_path:
                    doc = fitz.open(st.session_state.pdf_path)
                    pg = doc[c["page"]-1]
                    pix = pg.get_pixmap(matrix=fitz.Matrix(2,2))
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).copy()
                    if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    try:
                        bbox = ast.literal_eval(c.get("bbox_str", "(0,0,0,0)"))
                    except:
                        bbox = (0,0,0,0)
                    x0,y0,x1,y1 = bbox
                    if x0!=0 or y0!=0 or x1!=0 or y1!=0:
                        z = 2
                        cv2.rectangle(img, (int(x0*z),int(y0*z)), (int(x1*z),int(y1*z)), (0,255,0), 4)
                    st.image(img)
                    with st.expander("📜 证据原文", expanded=False):
                        st.text(c["content"][:400])
                    doc.close()
            st.divider()